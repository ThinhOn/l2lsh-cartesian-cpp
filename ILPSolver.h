// ilp_solver_cpp.h

#pragma once
#include "ortools/linear_solver/linear_solver.h"
#include "utils.h"

#include <string>
#include <vector>
#include <unordered_map>
#include <iostream>

class ILPSolverCpp {
public:
    // struct Candidate {
    //     std::string meta;   // same idea as your Python metadata string
    //     double distance;    // cost in the objective
    // };

    // query.count[attr][value] = required count
    // struct Query {
    //     int k;  // top-k
    //     std::unordered_map<
    //         std::string,
    //         std::unordered_map<std::string, int>
    //     > count;
    // };

    // Result mirrors Python: indices, objective, count, gaps
    // struct Result {
    //     using CountMap =
    //         std::unordered_map<
    //             std::string,
    //             std::unordered_map<std::string, int>
    //         >;

    //     std::vector<int> indices;
    //     double objective = 0.0;
    //     CountMap count;  // actual counts in selected set
    //     CountMap gaps;   // gap = need - actual
    // };

    ILPSolverCpp(
        bool msg = false,
        int time_limit_ms = -1,          // <0 means no limit
        bool allow_soft_counts = true,
        double soft_penalty_weight = 100.0
    )
        : msg_(msg),
          time_limit_ms_(time_limit_ms),
          allow_soft_counts_(allow_soft_counts),
          soft_penalty_weight_(soft_penalty_weight) {}

    SearchResult solve(const std::vector<Candidate>& candidates, const Query& query);

private:
    bool msg_;
    int time_limit_ms_;
    bool allow_soft_counts_;
    double soft_penalty_weight_;

    using AttrIndex =
        std::unordered_map<
            std::string,
            std::unordered_map<std::string, std::vector<int>>
        >;

    static AttrIndex build_attr_index(const std::vector<ParsedMeta>& metas);

    static std::unordered_map<std::string, int>
    count_attr(const std::string& attr,
               const std::vector<int>& indices,
               const std::vector<ParsedMeta>& metas);
};
