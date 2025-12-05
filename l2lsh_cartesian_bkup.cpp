#include <stag/data.h>
#include <stag/lsh.h>
#include "utils.h"

#include <Eigen/Dense>

#include <algorithm>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <fstream>


using namespace stag;


struct ParsedMeta {
    int id = -1;
    std::map<std::string, std::string> feats; // attribute -> value
};

// Parse strings like "id:0__gender:male__race:hispanic"
ParsedMeta parse_metadata_line(const std::string &s) {
    ParsedMeta pm;
    pm.id = -1;

    std::size_t pos = 0;
    while (pos < s.size()) {
        std::size_t next = s.find("__", pos);
        std::string token = (next == std::string::npos)
                            ? s.substr(pos)
                            : s.substr(pos, next - pos);

        std::size_t colon = token.find(':');
        if (colon != std::string::npos) {
            std::string key = token.substr(0, colon);
            std::string val = token.substr(colon + 1);

            std::transform(key.begin(), key.end(), key.begin(), ::tolower);
            std::transform(val.begin(), val.end(), val.begin(), ::tolower);

            if (key == "id") {
                pm.id = std::stoi(val);
            } else {
                pm.feats[key] = val;
            }
        }

        if (next == std::string::npos) break;
        pos = next + 2;
    }
    return pm;
}

using FeatureKey = std::vector<std::pair<std::string, std::string>>;

// Canonical feature key (ordered list of (attr,value))
FeatureKey make_feature_key(const std::map<std::string, std::string> &feats) {
    FeatureKey fk;
    fk.reserve(feats.size());
    for (auto &kv : feats) fk.push_back(kv); // map is already ordered by key
    return fk;
}

struct FeatureKeyHash {
    std::size_t operator()(FeatureKey const &fk) const noexcept {
        std::size_t h = 0;
        for (auto &kv : fk) {
            h ^= std::hash<std::string>()(kv.first)  + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<std::string>()(kv.second) + 0x9e3779b9 + (h << 6) + (h >> 2);
        }
        return h;
    }
};

// -------------------- per-partition structure --------------------

struct PartitionLSH {
    std::vector<int> ids;                 // row indices in global DenseMat
    std::vector<DataPoint> points;        // DataPoints referencing rows in data_
    std::unique_ptr<E2LSH> index;         // L2 LSH index
    std::unordered_map<const StagReal*, int> ptr2id; // coordinates pointer -> id
};

// -------------------- header --------------------

class L2LSHCartesianCpp {
public:
    L2LSHCartesianCpp(
        DenseMat &data,
        const std::vector<std::string> &metadata_store,
        const std::vector<std::string> &protected_attrs,
        StagUInt K,
        StagUInt L
    )
        : data_(data),
          metadata_store_(metadata_store),
          protected_attrs_(protected_attrs),
          K_(K),
          L_(L)
    {
        build_partitions();
        build_indices();
    }

    // Query a specific partition by name (e.g. "gender:female__race:hispanic")
    // and a query vector q_vec.
    std::vector<std::pair<int, double>>
    query_partition(const std::string &partition_name,
                    const Eigen::VectorXd &q_vec,
                    std::size_t max_results = 50) const;

    std::vector<std::string> partition_names() const {
        std::vector<std::string> names;
        names.reserve(partitions_.size());
        for (auto &kv : partitions_) names.push_back(kv.first);
        return names;
    }

private:
    DenseMat &data_;
    std::vector<std::string> metadata_store_;
    std::vector<std::string> protected_attrs_;
    StagUInt K_;
    StagUInt L_;

    // partition name -> data
    std::unordered_map<std::string, PartitionLSH> partitions_;

    void build_partitions();
    void build_indices();
};

// -------------------- L2LSHCartesianCpp implementation --------------------

// Build cartesian partitions from protected_attrs (e.g. ["gender:male", "gender:female", "race:hispanic", "race:asian"])
void L2LSHCartesianCpp::build_partitions() {
    // Group protected attrs by attribute name.
    // E.g. "gender:male" and "gender:female" go in same group.
    std::unordered_map<std::string, std::vector<std::string>> groups;
    for (auto &a : protected_attrs_) {
        std::size_t colon = a.find(':');
        if (colon == std::string::npos) continue;
        std::string key = a.substr(0, colon);
        groups[key].push_back(a);
    }
    if (groups.empty()) return;

    // To have a deterministic product, collect value lists.
    std::vector<std::vector<std::string>> group_lists;
    group_lists.reserve(groups.size());
    for (auto &kv : groups) {
        group_lists.push_back(kv.second);
    }

    // Cartesian product over groups.
    std::vector<std::string> all_partition_names;
    std::vector<std::string> current;

    std::function<void(std::size_t)> dfs = [&](std::size_t idx) {
        if (idx == group_lists.size()) {
            // Build canonical partition name: sorted tokens joined by "__"
            auto tokens = current;
            std::sort(tokens.begin(), tokens.end());
            std::string name;
            for (std::size_t i = 0; i < tokens.size(); ++i) {
                if (i > 0) name += "__";
                name += tokens[i];
            }
            all_partition_names.push_back(name);
            return;
        }
        for (auto &v : group_lists[idx]) {
            current.push_back(v);
            dfs(idx + 1);
            current.pop_back();
        }
    };
    dfs(0);

    // Map from feature key -> partition name
    std::unordered_map<FeatureKey, std::string, FeatureKeyHash> features_to_partition;
    for (auto &pname : all_partition_names) {
        ParsedMeta pm = parse_metadata_line(pname);  // only attrs, no id
        FeatureKey fk = make_feature_key(pm.feats);
        features_to_partition[fk] = pname;
        partitions_.emplace(pname, PartitionLSH{});
    }

    // Assign each metadata row to its partition (if it matches exactly)
    for (auto &line : metadata_store_) {
        ParsedMeta pm = parse_metadata_line(line);
        if (pm.id < 0) continue;
        FeatureKey fk = make_feature_key(pm.feats);
        auto it = features_to_partition.find(fk);
        if (it != features_to_partition.end()) {
            const std::string &pname = it->second;
            partitions_[pname].ids.push_back(pm.id);
        }
    }

    // Drop empty partitions
    for (auto it = partitions_.begin(); it != partitions_.end();) {
        if (it->second.ids.empty()) {
            it = partitions_.erase(it);
        } else {
            ++it;
        }
    }
}

// Build per-partition E2LSH indices
void L2LSHCartesianCpp::build_indices() {
    for (auto &kv : partitions_) {
        const std::string &name = kv.first;
        PartitionLSH &part = kv.second;
        if (part.ids.empty()) continue;

        part.points.clear();
        part.points.reserve(part.ids.size());
        for (int id : part.ids) {
            // DataPoint referencing row 'id' of data_
            part.points.emplace_back(data_, static_cast<StagInt>(id));
        }

        // Build pointer -> id map for quick lookup
        part.ptr2id.clear();
        part.ptr2id.reserve(part.points.size());
        for (std::size_t i = 0; i < part.points.size(); ++i) {
            part.ptr2id[part.points[i].coordinates] = part.ids[i];
        }

        std::cout << "Building E2LSH for partition " << name
                  << " with " << part.ids.size() << " points (K="
                  << K_ << ", L=" << L_ << ")\n";

        part.index = std::make_unique<E2LSH>(K_, L_, part.points);
    }
}

// Query a specific partition by name.
std::vector<std::pair<int, double>>
L2LSHCartesianCpp::query_partition(const std::string &partition_name,
                                   const Eigen::VectorXd &q_vec,
                                   std::size_t max_results) const
{
    auto it = partitions_.find(partition_name);
    if (it == partitions_.end()) {
        return {};
    }
    const PartitionLSH &part = it->second;
    if (!part.index) return {};

    // 1. Copy Eigen vector into mutable buffer, then wrap as DataPoint
    std::vector<StagReal> qbuf(q_vec.size());
    for (int i = 0; i < q_vec.size(); ++i) {
        qbuf[static_cast<std::size_t>(i)] = q_vec[i];
    }
    DataPoint q_dp(static_cast<StagUInt>(qbuf.size()), qbuf.data());

    // 2. Get LSH candidate set
    std::vector<DataPoint> cands = part.index->get_near_neighbors(q_dp);

    // 3. Compute distances and map to ids
    std::vector<std::pair<int, double>> out;
    out.reserve(cands.size());

    for (auto &c : cands) {
        auto itid = part.ptr2id.find(c.coordinates);
        if (itid == part.ptr2id.end()) continue;

        // DataPoint::to_vector() returns std::vector<StagReal>, so wrap via Eigen::Map
        std::vector<StagReal> c_std = c.to_vector();
        std::vector<StagReal> q_std = q_dp.to_vector();

        Eigen::Map<const Eigen::VectorXd> c_vec(c_std.data(),
                                                static_cast<int>(c_std.size()));
        Eigen::Map<const Eigen::VectorXd> q_vec2(q_std.data(),
                                                 static_cast<int>(q_std.size()));

        double dist2 = (c_vec - q_vec2).squaredNorm();
        out.emplace_back(itid->second, dist2);
    }

    std::sort(out.begin(), out.end(),
              [](auto &a, auto &b) { return a.second < b.second; });

    if (out.size() > max_results) out.resize(max_results);
    return out;
}



int main(int argc, char** argv) {
    // DenseMat X(6, 3);
    // X << 0, 0, 0,
    //      1, 0, 0,
    //      0, 1, 0,
    //      0, 0, 1,
    //      5, 5, 5,
    //      4.9, 5.1, 5.0;

    // // Metadata matching the ids (rows)
    // std::vector<std::string> metadata = {
    //     "id:0__gender:male__race:hispanic",
    //     "id:1__gender:male__race:hispanic",
    //     "id:2__gender:female__race:hispanic",
    //     "id:3__gender:female__race:hispanic",
    //     "id:4__gender:male__race:asian",
    //     "id:5__gender:female__race:asian"
    // };

    std::string DATASET = argv[1];
    DenseMat X = load_vectors_txt("./data/" + DATASET + "/vectors.txt");
    std::vector<std::string> metadata = load_metadata_txt("./data/" + DATASET + "/metadata.txt");

    // std::cout << "X.row(0) = " << X.row(0) << "\n";
    std::cout << "metadata[0] = " << metadata[0] << "\n";

    
    // Protected attribute values for Cartesian product
    auto attr_set = get_all_protected_attributes(metadata);
    std::vector<std::string> protected_attrs(attr_set.begin(), attr_set.end());
    
    StagUInt K = 6;
    StagUInt L = 10;

    
    L2LSHCartesianCpp index(X, metadata, protected_attrs, K, L);
    
    StagReal w = 4.0f;
    StagReal c = 2.0f;
    std::cout << collision_probability(w, c) << "\n";
    std::exit(0);


    // TODO: load query file


    // Example query: partition "gender:female__race:hispanic", query = row 2
    // std::string part_name = "gender:female__race:hispanic";
    // Eigen::VectorXd q = X.row(2);

    // auto results = index.query_partition(part_name, q, 10);

    // std::cout << "Results for partition " << part_name << ":\n";
    // for (auto &pr : results) {
    //     std::cout << "  id = " << pr.first << "  dist2 = " << pr.second << "\n";
    // }

    return 0;
}
