#include "utils.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <stag/lsh.h>
#include <cmath>
#include <nlohmann/json.hpp>


using namespace stag;
using json = nlohmann::json;


DenseMat load_vectors_txt(const std::string &filename) {
    std::ifstream in(filename);
    if (!in) {
        throw std::runtime_error("Cannot open " + filename);
    }

    std::vector<std::vector<double>> rows;
    std::string line;

    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::vector<double> row;
        double x;
        while (ss >> x) {
            row.push_back(x);
        }
        if (!row.empty()) rows.push_back(std::move(row));
    }

    if (rows.empty()) {
        throw std::runtime_error("No data in " + filename);
    }

    std::size_t n = rows.size();
    std::size_t d = rows[0].size();
    DenseMat X(n, d);

    for (std::size_t i = 0; i < n; ++i) {
        if (rows[i].size() != d) {
            throw std::runtime_error("Inconsistent dimensions in " + filename);
        }
        for (std::size_t j = 0; j < d; ++j) {
            X(static_cast<int>(i), static_cast<int>(j)) = rows[i][j];
        }
    }
    return X;
}


std::vector<std::string> load_metadata_txt(const std::string &filename) {
    std::ifstream in(filename);
    if (!in) {
        throw std::runtime_error("Cannot open " + filename);
    }
    std::vector<std::string> metas;
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty())
            metas.push_back(line);
    }
    return metas;
}


std::vector<std::string> split_metadata_fields(const std::string& s) {
    std::vector<std::string> out;
    std::size_t pos = 0;

    while (pos < s.size()) {
        std::size_t next = s.find("__", pos);
        std::string token = (next == std::string::npos)
            ? s.substr(pos)
            : s.substr(pos, next - pos);

        if (!token.empty()) out.push_back(token);

        if (next == std::string::npos) break;
        pos = next + 2;
    }
    return out;
}


bool is_protected_field(const std::string& token) {
    return token.rfind("id:", 0) != 0;  // false if token starts with "id:"
}


std::unordered_set<std::string> get_all_protected_attributes(
    const std::vector<std::string>& metadata_store)
{
    std::unordered_set<std::string> result;

    for (const std::string& line : metadata_store) {
        // split into ["id:0", "gender:male", "age:50-59", "race:east asian"]
        auto tokens = split_metadata_fields(line);

        for (const std::string& token : tokens) {
            if (is_protected_field(token)) {
                result.insert(token);   // add, e.g., "gender:male"
            }
        }
    }

    return result;
}


float collision_probability(float w, float c) {
    float x = w/c;

    return 1
            - std::erfc(x / M_SQRT2)
            - M_2_SQRTPI / M_SQRT2 / x * (1 - std::exp(-x * x / 2.0));
};



void from_json(const json& j, Query& q) {
    q.search_term = j.at("search_term").get<std::string>();
    q.text_query = j.at("text_query").get<std::string>();
    q.k = j.at("k").get<int>();

    // Nested dict → unordered_map<string, unordered_map<string, int>>
    q.count = j.at("count").get<
        std::unordered_map<std::string, std::unordered_map<std::string, int>>
    >();

    // Embedding list → std::vector<float>
    q.vec = j.at("text_query_embedding").get<std::vector<float>>();

    // List of [string, float] → vector<pair<string, float>>
    q.ground_truth = j.at("ground_truth").get<
        std::vector<std::pair<std::string, float>>
    >();

    q.ground_truth_dist = j.at("ground_truth_dist").get<float>();
};


void print_query_count(const Query& q) {
    for (const auto& [attr, inner_map] : q.count) {
        std::cout << attr << ":\n";
        for (const auto& [token, value] : inner_map) {
            std::cout << "    " << token << " : " << value << "\n";
        }
    }
}


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