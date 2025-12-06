#include "ILPSolver.h"
#include "utils.h"

using operations_research::MPSolver;
using operations_research::MPVariable;
using operations_research::MPConstraint;
using operations_research::MPObjective;

// ILPSolverCpp::Meta ILPSolverCpp::parse_meta(const std::string& s) {
//     Meta out;
//     std::size_t start = 0;
//     while (start < s.size()) {
//         std::size_t pos = s.find("__", start);
//         std::string part = s.substr(start,
//                                     (pos == std::string::npos
//                                          ? std::string::npos
//                                          : pos - start));
//         if (!part.empty()) {
//             std::size_t colon = part.find(':');
//             if (colon != std::string::npos) {
//                 std::string k = part.substr(0, colon);
//                 std::string v = part.substr(colon + 1);
//                 // you can add trimming here if needed
//                 out[k] = v;
//             }
//         }
//         if (pos == std::string::npos) break;
//         start = pos + 2; // skip "__"
//     }
//     return out;
// }

ILPSolverCpp::AttrIndex
ILPSolverCpp::build_attr_index(const std::vector<ParsedMeta>& metas) {
    AttrIndex idxs;
    const int n = static_cast<int>(metas.size());
    for (int i = 0; i < n; ++i) {
        for (const auto& kv : metas[i].feats) {
            const std::string& attr = kv.first;
            const std::string& val  = kv.second;
            if (attr == "id") continue;  // mimic your Python filter
            idxs[attr][val].push_back(i);
        }
    }
    return idxs;
}

std::unordered_map<std::string, int>
ILPSolverCpp::count_attr(const std::string& attr,
                         const std::vector<int>& indices,
                         const std::vector<Meta>& metas) {
    std::unordered_map<std::string, int> out;
    for (int idx : indices) {
        auto it = metas[idx].find(attr);
        if (it != metas[idx].end()) {
            out[it->second] += 1;
        }
    }
    return out;
}

// ---------- main solve() ----------

ILPSolverCpp::Result
ILPSolverCpp::solve(const std::vector<Candidate>& candidates,
                    const Query& query) {
    Result result;

    const int n = static_cast<int>(candidates.size());
    const int topk = query.k;

    // parse metadata strings to maps
    std::vector<Meta> metas;
    metas.reserve(n);
    for (const auto& c : candidates) {
        metas.push_back(parse_metadata_line(c.meta));
    }
    AttrIndex attr_index = build_attr_index(metas);

    // create solver
    MPSolver solver("select_k",
                    MPSolver::CBC_MIXED_INTEGER_PROGRAMMING);
    if (!msg_) {
        solver.SuppressOutput();
    } else {
        solver.EnableOutput();
    }

    if (time_limit_ms_ > 0) {
        solver.set_time_limit(time_limit_ms_);
    }

    const double inf = MPSolver::infinity();

    // x[i] in {0,1}
    std::vector<MPVariable*> x(n);
    for (int i = 0; i < n; ++i) {
        x[i] = solver.MakeIntVar(0.0, 1.0, "x_" + std::to_string(i));
    }

    // objective: sum distance_i * x_i
    MPObjective* objective = solver.MutableObjective();
    for (int i = 0; i < n; ++i) {
        objective->SetCoefficient(x[i], candidates[i].distance);
    }
    objective->SetMinimization();

    // cardinality: sum x_i == topk
    {
        MPConstraint* c = solver.MakeRowConstraint(topk, topk, "choose_topk");
        for (int i = 0; i < n; ++i) {
            c->SetCoefficient(x[i], 1.0);
        }
    }

    // build constraints (hard exact or soft)
    if (!allow_soft_counts_) {
        // ----- exact constraints branch (mirror _add_exact_constraints) -----
        for (const auto& attr_entry : query.count) {
            const std::string& attr = attr_entry.first;
            const auto& vm = attr_entry.second;

            int specified_total = 0;
            for (const auto& kv : vm) {
                const std::string& val = kv.first;
                int need = kv.second;
                specified_total += need;

                // indices with this (attr, val)
                std::vector<int> rows;
                auto it_attr = attr_index.find(attr);
                if (it_attr != attr_index.end()) {
                    auto it_val = it_attr->second.find(val);
                    if (it_val != it_attr->second.end()) {
                        rows = it_val->second;
                    }
                }

                MPConstraint* c =
                    solver.MakeRowConstraint(need, need,
                                             "eq_" + attr + "_" + val);
                for (int idx : rows) {
                    c->SetCoefficient(x[idx], 1.0);
                }
            }

            // if sums to topk, forbid all other values for that attr
            if (specified_total == topk) {
                std::vector<char> covered(n, false);
                auto it_attr = attr_index.find(attr);
                if (it_attr != attr_index.end()) {
                    for (const auto& kv : vm) {
                        const std::string& val = kv.first;
                        auto it_val = it_attr->second.find(val);
                        if (it_val != it_attr->second.end()) {
                            for (int idx : it_val->second) {
                                covered[idx] = true;
                            }
                        }
                    }
                }
                std::vector<int> other;
                for (int i = 0; i < n; ++i) {
                    if (!covered[i]) other.push_back(i);
                }
                if (!other.empty()) {
                    MPConstraint* c =
                        solver.MakeRowConstraint(0.0, 0.0,
                                                 "forbid_other_" + attr);
                    for (int idx : other) {
                        c->SetCoefficient(x[idx], 1.0);
                    }
                }
            }
        }
    } else {
        // ----- soft exact constraints branch (mirror _add_soft_exact_constraints) -----
        for (const auto& attr_entry : query.count) {
            const std::string& attr = attr_entry.first;
            const auto& vm = attr_entry.second;

            for (const auto& kv : vm) {
                const std::string& val = kv.first;
                int need = kv.second;

                // indices with this (attr, val)
                std::vector<int> rows;
                auto it_attr = attr_index.find(attr);
                if (it_attr != attr_index.end()) {
                    auto it_val = it_attr->second.find(val);
                    if (it_val != it_attr->second.end()) {
                        rows = it_val->second;
                    }
                }

                const std::string base_name = attr + "_" + val;
                MPVariable* over =
                    solver.MakeNumVar(0.0, inf, "over_" + base_name);
                MPVariable* under =
                    solver.MakeNumVar(0.0, inf, "under_" + base_name);

                // sum rows x_i - over + under = need
                MPConstraint* c =
                    solver.MakeRowConstraint(need, need,
                                             "soft_" + base_name);
                for (int idx : rows) {
                    c->SetCoefficient(x[idx], 1.0);
                }
                c->SetCoefficient(over, -1.0);
                c->SetCoefficient(under, 1.0);

                // penalty in objective
                objective->SetCoefficient(
                    over,
                    objective->GetCoefficient(over) + soft_penalty_weight_);
                objective->SetCoefficient(
                    under,
                    objective->GetCoefficient(under) + soft_penalty_weight_);
            }
        }
    }

    // ---------- solve ----------
    const MPSolver::ResultStatus status = solver.Solve();

    if (status != MPSolver::OPTIMAL &&
        status != MPSolver::FEASIBLE) {
        if (msg_) {
            std::cerr << "ILPSolverCpp: no feasible solution, status = "
                      << status << "\n";
        }
        return result; // empty result
    }

    // selected indices
    for (int i = 0; i < n; ++i) {
        if (x[i]->solution_value() > 0.5) {
            result.indices.push_back(i);
        }
    }

    result.objective = objective->Value();

    // ---------- counts & gaps (Python-style post-processing) ----------
    for (const auto& attr_entry : query.count) {
        const std::string& attr = attr_entry.first;
        // actual counts for this attr among selected indices
        auto actual = count_attr(attr, result.indices, metas);
        result.count[attr] = actual;

        // gap = need - actual (if not present in actual, treat as 0)
        for (const auto& kv : attr_entry.second) {
            const std::string& val = kv.first;
            int need = kv.second;
            int got = 0;
            auto it_val = actual.find(val);
            if (it_val != actual.end()) {
                got = it_val->second;
            }
            result.gaps[attr][val] = need - got;
        }
    }

    return result;
}
