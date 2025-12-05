#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <string>
#include <unordered_set>
#include <utility>

#include <stag/data.h>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

DenseMat load_vectors_txt(const std::string &filename);

std::vector<std::string> load_metadata_txt(const std::string &filename);

std::vector<std::string> split_metadata_fields(const std::string& s);
std::unordered_set<std::string> get_all_protected_attributes(
    const std::vector<std::string>& metadata_store
);
bool is_protected_field(const std::string& token);

float collision_probability(float w, float c);


struct Query {
    std::string search_term;
    std::string text_query;
    int k;

    std::unordered_map<std::string,
                       std::unordered_map<std::string, int>> count;

    std::vector<float> vec;

    std::vector<std::pair<std::string, float>> ground_truth;

    float ground_truth_dist;
};

void from_json(const json& j, Query& s);
void print_query_count(const Query& q);


struct SearchResult {
    std::vector<std::pair<std::string, float>> chosen;
    double search_time_ms = 0.0;
    double post_time_ms   = 0.0;
};

#endif