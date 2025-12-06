void SolveILP(
    const std::vector<Candidate>& candidates,
    const Query& query,
    bool allow_soft_counts = false,
    double soft_penalty_weight = 100.0)
{
    const int n = (int)candidates.size();
    const int topk = query.k;

    // --- Parse metas and build attr index ---
    std::vector<Meta> metas;
    metas.reserve(n);
    for (const auto& c : candidates) {
        metas.push_back(ParseMeta(c.meta));
    }
    AttrIndex attr_index = BuildAttrIndex(metas);

    // --- Create solver (CBC MILP) ---
    MPSolver solver("select_k",
        MPSolver::CBC_MIXED_INTEGER_PROGRAMMING);

    const double inf = solver.infinity();

    // --- Decision vars: x[i] in {0,1} ---
    std::vector<MPVariable*> x(n);
    for (int i = 0; i < n; ++i) {
        x[i] = solver.MakeIntVar(0.0, 1.0, "x_" + std::to_string(i));
    }

    // --- Objective: minimize sum distance_i * x_i ---
    MPObjective* objective = solver.MutableObjective();
    for (int i = 0; i < n; ++i) {
        objective->SetCoefficient(x[i], candidates[i].distance);
    }
    objective->SetMinimization();

    // --- Cardinality: sum x_i == topk ---
    {
        MPConstraint* c = solver.MakeRowConstraint(topk, topk, "choose_topk");
        for (int i = 0; i < n; ++i) {
            c->SetCoefficient(x[i], 1.0);
        }
    }

    // --- Add exact or soft count constraints ---
    if (!allow_soft_counts) {
        // Hard constraints: direct OR-Tools version of _add_exact_constraints
        for (const auto& [attr, vm] : query.counts) {
            int specified_total = 0;
            for (const auto& [val, need] : vm) {
                specified_total += need;

                auto it_attr = attr_index.find(attr);
                std::vector<int> rows;
                if (it_attr != attr_index.end()) {
                    auto it_val = it_attr->second.find(val);
                    if (it_val != it_attr->second.end())
                        rows = it_val->second;
                }

                MPConstraint* c = solver.MakeRowConstraint(need, need);
                for (int idx : rows) {
                    c->SetCoefficient(x[idx], 1.0);
                }
            }

            // forbid other values if specified_total == topk
            if (specified_total == topk) {
                // collect indices covered by any specified val
                std::vector<char> covered(n, false);
                const auto& vm_attr = vm;
                auto it_attr = attr_index.find(attr);
                if (it_attr != attr_index.end()) {
                    for (const auto& [val, need] : vm_attr) {
                        auto it_val = it_attr->second.find(val);
                        if (it_val != it_attr->second.end()) {
                            for (int idx : it_val->second) {
                                covered[idx] = true;
                            }
                        }
                    }
                }
                // other indices must be 0
                std::vector<int> other;
                for (int i = 0; i < n; ++i) {
                    if (!covered[i]) other.push_back(i);
                }
                if (!other.empty()) {
                    MPConstraint* c = solver.MakeRowConstraint(0.0, 0.0);
                    for (int idx : other) {
                        c->SetCoefficient(x[idx], 1.0);
                    }
                }
            }
        }
    } else {
        // --- Soft version: translate _add_soft_exact_constraints ---
        // For each (attr, val, need):
        //    sum rows x_i - over + under = need
        // and objective += soft_penalty_weight * (over + under).
        for (const auto& [attr, vm] : query.counts) {
            for (const auto& [val, need] : vm) {
                auto it_attr = attr_index.find(attr);
                std::vector<int> rows;
                if (it_attr != attr_index.end()) {
                    auto it_val = it_attr->second.find(val);
                    if (it_val != it_attr->second.end())
                        rows = it_val->second;
                }

                // slack vars
                std::string base_name = attr + "_" + val;
                MPVariable* over = solver.MakeNumVar(0.0, inf, "over_" + base_name);
                MPVariable* under = solver.MakeNumVar(0.0, inf, "under_" + base_name);

                MPConstraint* c = solver.MakeRowConstraint(need, need);
                for (int idx : rows) {
                    c->SetCoefficient(x[idx], 1.0);
                }
                c->SetCoefficient(over, -1.0);
                c->SetCoefficient(under, 1.0);

                // add penalty to objective
                objective->SetCoefficient(over,
                    objective->GetCoefficient(over) + soft_penalty_weight);
                objective->SetCoefficient(under,
                    objective->GetCoefficient(under) + soft_penalty_weight);
            }
        }
    }

    // --- Solve ---
    const MPSolver::ResultStatus status = solver.Solve();

    if (status != MPSolver::OPTIMAL &&
        status != MPSolver::FEASIBLE) {
        std::cerr << "No feasible solution, status = " << status << "\n";
        return;
    }

    // --- Extract indices (like Python: pl.value(x[i]) > 0.5) ---
    std::vector<int> indices;
    for (int i = 0; i < n; ++i) {
        if (x[i]->solution_value() > 0.5) {
            indices.push_back(i);
        }
    }

    std::cout << "Objective = " << objective->Value() << "\n";
    std::cout << "Selected indices: ";
    for (int idx : indices) std::cout << idx << " ";
    std::cout << "\n";
}
