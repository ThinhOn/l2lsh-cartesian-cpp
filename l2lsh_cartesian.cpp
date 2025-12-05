#include <stag/data.h>
#include <stag/lsh.h>
#include "utils.h"

#include <Eigen/Dense>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <fstream>


using namespace stag;
using json = nlohmann::json;

using AttrCountMap = std::unordered_map<std::string, int>;
using CountMap     = std::unordered_map<std::string, AttrCountMap>;
using Vec = std::vector<float>;


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
        const float c = 2.0f,
        const float r = 2.0f,
        const float w = 4.0f,
        const int max_query_size = 20
    )
        : data_(data),
          metadata_store_(metadata_store),
          protected_attrs_(protected_attrs),
          c_(c),
          r_(r),
          w_(w)
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

    std::vector<std::pair<int, double>>
    L2LSHCartesianCpp::search(
        const Eigen::VectorXd& qvec,
        const CountMap& constraints,
        DenseMat& vector_store
    ) const;

private:
    DenseMat &data_;
    std::vector<std::string> metadata_store_;
    std::vector<std::string> protected_attrs_;
    StagReal c_;
    StagReal r_;
    StagReal w_;
    int max_query_size = 20;  // number of hash functions per table

    // partition name -> data
    std::unordered_map<std::string, PartitionLSH> partitions_;

    void build_partitions();
    void build_indices();
};

// -------------------- L2LSHCartesianCpp implementation --------------------

// Build cartesian partitions from protected_attrs (e.g. ["gender:male", "gender:female", "race:hispanic", "race:asian"])
void L2LSHCartesianCpp::build_partitions() {
    // group protected attrs by attribute name.
    // e.g. "gender:male" and "gender:female" go in same group.
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
            // Build partition name: sorted tokens joined by "__"
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

        // std::cout << "Building E2LSH for partition " << name
        //           << " with " << part.ids.size() << " points (K="
        //           << K_ << ", L=" << L_ << ")\n";
        
        int K_=10;
        int L_=5;
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

// Search on all relevant partitions
// step 1: retrieve near points
// step 2: pick k_star
// step 3: sort by distance
// step 4: return k_pi nearest points
SearchResult L2LSHCartesianCPP::search(
    const Eigen::VectorXd& qvec,
    const CountMap& constraints,
    DenseMat& vector_store
) const {
    using Clock = std::chrono::high_resolution_clock;

    SearchResult result;
    std::size_t& total_scan_out = 0;

    // ---------- 1) Build Cartesian product of constraint tokens ----------
    // constraints: attr -> { value -> count }
    // lists: [[ "gender:female", "gender:male" ], [ "age:50-59", "age:60-69", ...], ...]
    std::vector<std::vector<std::string>> lists;
    lists.reserve(constraints.size());
    for (const auto& [attr, values] : constraints) {
        std::vector<std::string> tokens_for_attr;
        tokens_for_attr.reserve(values.size());
        for (const auto& [value, _cnt] : values) {
            tokens_for_attr.push_back(attr + ":" + value);
        }
        lists.push_back(std::move(tokens_for_attr));
    }

    // recursive Cartesian product
    std::vector<std::vector<std::string>> combos;
    {
        std::vector<std::string> current;
        std::function<void(std::size_t)> dfs = [&](std::size_t idx) {
            if (idx == lists.size()) {
                combos.push_back(current);
                return;
            }
            for (const auto& token : lists[idx]) {
                current.push_back(token);
                dfs(idx + 1);
                current.pop_back();
            }
        };
        dfs(0);
    }

    struct ComboInfo {
        int requirement;
        std::vector<std::string> partitions;
    };

    std::unordered_map<std::vector<std::string>, ComboInfo,
                       HashKeyHash> combo_info; // reuse HashKeyHash for vectors of strings if you want; or write separate hasher

    auto start_search = Clock::now();

    // ---------- 2) Compute requirement & matching partitions per combo ----------
    for (const auto& combo : combos) {
        // requirement = min(counts[attr][val] for token in combo)
        int requirement = std::numeric_limits<int>::max();
        for (const auto& token : combo) {
            auto pos = token.find(':');
            std::string attr = token.substr(0, pos);
            std::string value = token.substr(pos + 1);
            auto it_attr = constraints.find(attr);
            if (it_attr == constraints.end()) continue;
            auto it_val = it_attr->second.find(value);
            if (it_val == it_attr->second.end()) continue;
            requirement = std::min(requirement, it_val->second);
        }
        if (requirement == std::numeric_limits<int>::max()) {
            // no valid counts; skip
            continue;
        }

        // find partitions that contain all tokens in combo
        std::unordered_set<std::string> combo_set(combo.begin(), combo.end());
        std::vector<std::string> matching_parts;
        for (const auto& [pname, tokens] : partition_tokens) {
            bool ok = true;
            for (const auto& t : combo_set) {
                if (!tokens.count(t)) {
                    ok = false;
                    break;
                }
            }
            if (ok) matching_parts.push_back(pname);
        }

        if (!matching_parts.empty()) {
            combo_info[combo] = ComboInfo{requirement, std::move(matching_parts)};
        }
    }

    // ---------- 3) For each combo, gather ANN candidates ----------
    std::vector<std::pair<int, float>> final_cands; // (id, dist)
    final_cands.reserve(1024);

    for (const auto& [combo, info] : combo_info) {
        int k_pi = info.requirement;
        std::vector<int> all_cands;

        for (const auto& pi : info.partitions) {
            auto tables_it = tables.find(pi);
            auto hashes_it = hashes.find(pi);
            if (tables_it == tables.end() || hashes_it == hashes.end())
                continue;

            const TableList& T_list = tables_it->second;
            const auto& H_list      = hashes_it->second;
            std::size_t ell         = T_list.size();

            int k_star = k_pi + static_cast<int>(std::ceil(2.0 * ell / delta));

            std::vector<int> cands;

            // For each table in this partition
            for (std::size_t j = 0; j < ell; ++j) {
                const BucketMap& T = T_list[j];
                const auto& g      = H_list[j];  // your CompoundHash equivalent

                HashKey key = g.hash(qvec);      // or g(qvec) if operator() implemented

                auto bit = T.find(key);
                if (bit != T.end()) {
                    const auto& bucket_ids = bit->second;
                    cands.insert(cands.end(), bucket_ids.begin(), bucket_ids.end());
                }
            }

            // deduplicate
            std::sort(cands.begin(), cands.end());
            cands.erase(std::unique(cands.begin(), cands.end()), cands.end());

            if (static_cast<int>(cands.size()) > k_star) {
                cands.resize(k_star);
            }

            total_scan_out += cands.size();
            all_cands.insert(all_cands.end(), cands.begin(), cands.end());
        }

        // deduplicate across all partitions
        std::sort(all_cands.begin(), all_cands.end());
        all_cands.erase(std::unique(all_cands.begin(), all_cands.end()), all_cands.end());

        // compute distances
        std::vector<std::pair<int, float>> pairs;
        pairs.reserve(all_cands.size());
        for (int id : all_cands) {
            float dist = l2_distance(qvec, vector_store[id]);
            pairs.emplace_back(id, dist);
        }

        // sort by distance
        std::sort(pairs.begin(), pairs.end(),
                  [](auto const& a, auto const& b) { return a.second < b.second; });

        // top k_pi
        if (pairs.size() > static_cast<std::size_t>(k_pi))
            pairs.resize(k_pi);

        final_cands.insert(final_cands.end(), pairs.begin(), pairs.end());
    }

    auto end_search = Clock::now();
    result.search_time_ms =
        std::chrono::duration<double, std::milli>(end_search - start_search).count();

    // Map IDs to metadata strings, like Python’s:
    // final_cands = [(metadata_store[cand[0]], cand[1]) ...]
    std::vector<std::pair<std::string, float>> final_cands_str;
    final_cands_str.reserve(final_cands.size());
    for (auto const& [id, dist] : final_cands) {
        if (id >= 0 && static_cast<std::size_t>(id) < metadata_store.size()) {
            final_cands_str.emplace_back(metadata_store[id], dist);
        }
    }

    if (final_cands_str.empty()) {
        return result; // nothing found; solver would return None in Python
    }

    // ---------- 4) Post-processing: call solver (TODO: your C++ solver) ----------
    auto start_post = Clock::now();

    // TODO: call your C++ solver equivalent of `build_solver(self.args).solve(...)`
    // Example:
    //   Solver solver(args);
    //   auto solved = solver.solve(final_cands_str, constraints_or_query);
    // For now, just pass through:
    result.chosen = std::move(final_cands_str);

    auto end_post = Clock::now();
    result.post_time_ms =
        std::chrono::duration<double, std::milli>(end_post - start_post).count();

    return result;
}



int main(int argc, char** argv) {

    std::string DATASET = argv[1];
    float c = std::stof(argv[2]);
    float r = std::stof(argv[3]);
    float w = std::stof(argv[4]);

    DenseMat X = load_vectors_txt("./data/" + DATASET + "/vectors.txt");
    std::vector<std::string> metadata = load_metadata_txt("./data/" + DATASET + "/metadata.txt");

    // std::cout << "X.row(0) = " << X.row(0) << "\n";
    std::cout << "metadata[0] = " << metadata[0] << "\n";

    
    // Protected attribute values for Cartesian product
    auto attr_set = get_all_protected_attributes(metadata);
    std::vector<std::string> protected_attrs(attr_set.begin(), attr_set.end());

    
    L2LSHCartesianCpp index(
        X,
        metadata,
        protected_attrs,
        c,
        r,
        w
    );
    
    // std::cout << collision_probability(w, r) << "\n";
    // std::exit(0);


    // Load query file
    std::ifstream f(
        "./data/"
        + DATASET
        + "/queries_complex_k=[5,8,10,15,20].json"
    );
    json j;
    f >> j;

    std::vector<Query> queries = j.get<std::vector<Query>>();
    std::cout << "Loaded " << queries.size() << " queries.\n";

    for (auto &q : queries) {
        std::cout << "  search_term: " << q.search_term
                  << "  text_query_embedding size: "
                  << q.text_query_embedding.size() << "\n";
        Eigen::VectorXd q_vec = Eigen::Map<const Eigen::VectorXf>(q.text_query_embedding.data(), q.text_query_embedding.size()).cast<float>();
        index.search(q_vec, q.count, X);
        std::exit(0);
    }
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
