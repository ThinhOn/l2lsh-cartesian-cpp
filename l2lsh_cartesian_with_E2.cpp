#include <stag/data.h>
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

// ---- E2LSH headers ----
// Adjust these includes depending on where you installed E2LSH.
// If you copied headers into /usr/local/include/E2LSH, you can use:
//   #include <E2LSH/BasicDefinitions.h>
//   #include <E2LSH/Geometry.h>
//   #include <E2LSH/NearNeighbors.h>
#include "BasicDefinitions.h"
#include "Geometry.h"
#include "NearNeighbors.h"

using namespace stag;

// -------------------- metadata parsing --------------------

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

// This struct is now E2LSH-native.
struct PartitionLSH {
    std::vector<int> ids;
    std::vector<PPointT> points;
    std::vector<std::vector<float>> coords;
    PRNearNeighborStructT index;
    ~PartitionLSH() { for (auto p : points) delete p; }
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

    // Query a specific partition by name and return up to max_results
    // (id, squared_distance) pairs.
    std::vector<std::pair<int, double>>
    query_partition(const std::string &partition_name,
                    const Eigen::VectorXd &q_vec,
                    std::size_t max_results) const;

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

// Build cartesian partitions from protected_attrs (e.g. ["gender:male",
// "gender:female", "race:hispanic", "race:asian"])
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

    // Recursive cartesian product over the groups
    std::vector<std::string> all_partition_names;

    std::function<void(std::size_t, std::map<std::string, std::string>)> rec;
    rec = [&](std::size_t idx, std::map<std::string, std::string> cur) {
        if (idx == group_lists.size()) {
            // create partition name "gender:female__race:asian", etc.
            std::string name;
            bool first = true;
            for (auto &kv : cur) {
                if (!first) name += "__";
                first = false;
                name += kv.first + ":" + kv.second;
            }
            all_partition_names.push_back(name);
            return;
        }
        for (auto &val : group_lists[idx]) {
            std::size_t colon = val.find(':');
            if (colon == std::string::npos) continue;
            std::string key = val.substr(0, colon);
            std::string v   = val.substr(colon + 1);
            auto cur2 = cur;
            cur2[key] = v;
            rec(idx + 1, std::move(cur2));
        }
    };

    rec(0, std::map<std::string, std::string>{});

    // Map: combination of features -> partition name
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

// Build per-partition E2LSH indices (original E2LSH library)
void L2LSHCartesianCpp::build_indices() {
    // Initialize global E2LSH state once.
    initializeLSHGlobal();

    // LSH parameters you may want to tune:
    const float R = 1.0;          // search radius
    const float successProb = 0.9;
    const MemVarT memoryUpperBound = static_cast<MemVarT>(200000000); // ~200MB
    const IntT dim = static_cast<IntT>(data_.cols());

    for (auto &kv : partitions_) {
        const std::string &name = kv.first;
        PartitionLSH &part = kv.second;

        if (part.ids.empty()) continue;

        const Int32T nPoints = static_cast<Int32T>(part.ids.size());

        // Allocate storage for coordinates and points
        part.coords.clear();
        part.points.clear();
        part.coords.resize(nPoints);
        part.points.resize(nPoints);

        for (Int32T i = 0; i < nPoints; ++i) {
            int row_id = part.ids[i];

            // Copy row from DenseMat into float buffer
            part.coords[i].resize(dim);
            for (IntT d = 0; d < dim; ++d) {
                part.coords[i][d] = static_cast<float>(data_(row_id, d));
            }

            // Allocate E2LSH point struct
            PPointT p = new PointT();
            p->index = row_id;
            p->coordinates = part.coords[i].data();

            // Compute squared length
            float norm2 = 0;
            for (IntT d = 0; d < dim; ++d) {
                float v = p->coordinates[d];
                norm2 += v * v;
            }
            p->sqrLength = norm2;

            part.points[i] = p;
        }

        // Use a subset of points as sample queries for tuning
        IntT nSampleQueries = std::min<IntT>(nPoints, 50);
        std::vector<PPointT> sampleQueries(nSampleQueries);
        for (IntT i = 0; i < nSampleQueries; ++i) {
            sampleQueries[i] = part.points[i];
        }

        std::cout << "Building E2LSH (MIT) for partition " << name
                  << " with " << nPoints << " points (dim=" << dim << ")\n";

        // Build self-tuned structure
        part.index = initSelfTunedRNearNeighborWithDataSet(
            R,
            successProb,
            nPoints,
            dim,
            part.points.data(),
            nSampleQueries,
            sampleQueries.data(),
            memoryUpperBound
        );
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

    const IntT dim = static_cast<IntT>(q_vec.size());

    // 1. Wrap Eigen vector as a temporary E2LSH point
    std::vector<float> qbuf(dim);
    for (IntT d = 0; d < dim; ++d) {
        qbuf[d] = static_cast<float>(q_vec[d]);
    }

    PPointT q = new PointT();
    q->coordinates = qbuf.data();

    float norm2 = 0;
    for (IntT d = 0; d < dim; ++d) {
        float v = q->coordinates[d];
        norm2 += v * v;
    }
    q->sqrLength = norm2;
    q->index = -1;  // query has no dataset index

    // 2. Call E2LSH
    PPointT *result = nullptr;
    Int32T resultSize = 0;

    Int32T nFound = getRNearNeighbors(part.index, q, result, resultSize);

    std::vector<std::pair<int, double>> out;
    out.reserve(static_cast<std::size_t>(nFound));

    // 3. Convert neighbors to (id, distance^2)
    for (Int32T i = 0; i < nFound; ++i) {
        PPointT p = result[i];
        int id = p->index;

        // Recompute squared distance from original matrix using Eigen
        Eigen::VectorXd x = data_.row(id);
        double dist2 = (x - q_vec).squaredNorm();

        out.emplace_back(id, dist2);
    }

    // Sort by distance
    std::sort(out.begin(), out.end(),
              [](const auto &a, const auto &b) { return a.second < b.second; });

    if (out.size() > max_results) out.resize(max_results);

    delete q;
    // Depending on your E2LSH version, result might need to be freed explicitly.
    // Check the original LSHMain.cpp to see if they do "free(result);" etc.

    return out;
}

// -------------------- main --------------------

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " DATASET_NAME\n";
        return 1;
    }

    std::string DATASET = argv[1];

    DenseMat X = load_vectors_txt("./data/" + DATASET + "/vectors.txt");
    std::vector<std::string> metadata = load_metadata_txt("./data/" + DATASET + "/metadata.txt");

    if (metadata.empty()) {
        std::cerr << "No metadata loaded.\n";
        return 1;
    }

    std::cout << "metadata[0] = " << metadata[0] << "\n";

    // Protected attribute values for Cartesian product
    auto attr_set = get_all_protected_attributes(metadata);
    std::vector<std::string> protected_attrs(attr_set.begin(), attr_set.end());

    StagUInt K = 6;
    StagUInt L = 10;

    L2LSHCartesianCpp index(X, metadata, protected_attrs, K, L);

    // TODO: wire up real queries here (e.g. from a file of query vectors)
    // Example:
    // std::string part_name = "gender:female__race:hispanic";
    // Eigen::VectorXd q = X.row(2);
    // auto results = index.query_partition(part_name, q, 10);
    // for (auto &pr : results) {
    //     std::cout << "id=" << pr.first << "  dist2=" << pr.second << "\n";
    // }

    return 0;
}
