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

struct Candidate {
    std::string meta;   // same idea as your Python metadata string
    double distance;    // cost in the objective
};

void from_json(const json& j, Query& s);
void print_query_count(const Query& q);


struct SearchResult {
    using CountMap =
        std::unordered_map<
            std::string,
            std::unordered_map<std::string, int>
        >;

    std::vector<int> indices;
    double objective = 0.0;
    CountMap count;  // actual counts in selected set
    CountMap gaps;   // gap = need - actual
    float search_time_ms;
    std::vector<std::pair<std::string, float>> chosen;
};

struct ParsedMeta {
    int id = -1;
    std::map<std::string, std::string> feats; // attribute -> value
};

ParsedMeta parse_metadata_line(const std::string &s);

#endif