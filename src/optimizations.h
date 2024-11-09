#ifndef __OPTIMIZATION_H__
#define __OPTIMIZATION_H__

namespace dg::optimization::path{

    //today let's brush up on the path knowledge - this is very important - probably the most important in optimization because every problem can be formalized as NN and path
    //proof - this I tried to prove 10 years ago but now I have finally formalized the proof
    //assume there exists the shortest path between any two point A and B, shortest(A, B) = shortest(A, intermediate) + shortest(intermediate, B) 
    //assume the current active nodes set of C
    //for every iteration, we try to remove one active node D by passing responsibily to other nodes c C - such that every node that is reachable through D is reachable through the remaining active nodes

    auto floyd_path(const std::unordered_set<size_t>& nodes, const std::unordered_map<std::pair<size_t, size_t>, double>& edge_value) -> std::unordered_map<std::pair<size_t, size_t>, double>{
        
        const double MAX_PATH_VALUE = size_t{1} << 30;
        std::unordered_map<std::pair<size_t, size_t>, double> rs{};
        
        for (size_t i = 0u; i < nodes.size(); ++i){
            for (size_t j = 0u; j < nodes.size(); ++j){
                rs[std::make_pair(i, j)] = MAX_PATH_VALUE;
            }
        }

        for (const auto& map_pair: edge_value){
            rs[map_pair.first] = map_pair.second;
        }

        for (size_t i = 0u; i < nodes.size(); ++i){
            for (size_t j = 0u; j < nodes.size(); ++j){
                for (size_t z = 0u; z < nodes.size(); ++z){
                    auto ab_path    = std::make_pair(j, z);
                    auto ac_path    = std::make_pair(j, i);
                    auto cb_path    = std::make_pair(i, z); 

                    rs[ab_path] = std::min(rs[ab_path], rs[ac_path] + rs[cb_path]);
                }
            }
        }

        return rs;
    }

    //node expansion - non-negative - lowest at a time
    //forget about the const for now - I never knew why std does not have const version for accessor - which is absurd - maybe prevent people from shooting their feet - but they allowed it in optional.value() - maybe that's semantics - idk

    auto dijkstra_sssp(size_t src_id, const std::unordered_map<size_t, std::vector<size_t>>& edges, const std::unordered_map<std::pair<size_t, size_t>, double>& edge_value) -> std::unordered_map<size_t, double>{

        std::unordered_set<size_t> visited_nodes{};
        std::unordered_map<size_t, double> rs{};
        std::vector<std::pair<size_t, double>> priority_queue{};

        priority_queue.push_back(std::make_pair(src_id, 0u));

        auto cmp = [](const auto& lhs, const auto& rhs){
            return lhs.second > rhs.second;
        };

        while (!priority_queue.empty()){
            std::pop_heap(priority_queue.begin(), priority_queue.end(), cmp);
            auto [hinge, dist] = priority_queue.back();
            priority_queue.pop_back();
            
            if (visited_nodes.contains(hinge)){
                continue;
            }
            
            visited_nodes.insert(hinge);
            rs[hinge] = dist;

            for (size_t i = 0u; i < edges[hinge].size(); ++i){
                size_t dst  = edges[hinge][i];
                double val  = dist + edge_value[std::make_pair(hinge, dst)];
                priority_queue.push_back(std::make_pair(dst, val));
                std::push_heap(priority_queue.begin(), priority_queue.end(), cmp);
            }
        }

        return rs;
    }

    //admissible heuristic - estimate(A, B) <= actual_shortest_path(A, B)
    //why? because assume that there is the shortest path between A and B - then the yet-to-be-expanded-path will be expanded - because the last_node of the not correct path that touches the destination will be of lower priority in the priority_queue - because this is where heuristic and actual difference approaches 0
    //then there is a question. What about is visited_node set?
    //is the logic of visited_node the same as dijkstra - such that the lowest at any given point (local_mimina(node)) in time is guaranteed to be the lowest globally (global_minima(node))?
    //proof:
    //assume that the heuristic is stable and admissible - such that estimate(A, B) is constant
    //assume that there is a not-yet-to-be-expanded path that reach A and results in a lower heuristic
    //assume that the current heuristic of an arbitrary D is greater than the current heuristic of the current A
    //call the shortest path between D and A - P
    //assume the shortest path value is non-negative
    //heuristic(D) + value(P) > heuristic(A) - contradiction
    //usually - astar is for finding the A - B path rather than sssp - so let's change it here

    template <class Heuristic>
    auto astar_sssp(size_t src_id, size_t dst_id, const std::unordered_map<size_t, std::vector<size_t>>& edges, const std::unordered_map<std::pair<size_t, size_t>, double>& edge_value, Heuristic&& heuristic) -> std::optional<double>{

        std::unordered_set<size_t> visited_nodes{};
        std::unordered_map<size_t, double> heuristic_map{};
        std::vector<std::pair<size_t, double>> priority_queue{};

        priority_queue.push_back(std::make_pair(src_id, heuristic(src_id, dst_id)));
        auto cmp = [](const auto& lhs, const auto& rhs){
            return lhs.second > rhs.second;
        };

        while (true){
            if (priority_queue.empty()){
                return std::nullopt;
            }

            std::pop_heap(priority_queue.begin(), priority_queue.end(), cmp);
            auto [hinge, dist] = priority_queue.back();
            priority_queue.pop_back(); 

            if (hinge == dst_id){
                return dist;
            }

            if (visited_nodes.contains(hinge)){
                continue;
            }

            visited_nodes.insert(hinge);
            heuristic_map[hinge] = dist; 

            for (size_t i = 0u; i < edges[hinge].size(); ++i){
                size_t neighbor         = edges[hinge][i];
                double new_heuristic    = dist - heuristic(hinge, dst_id) + edge_value[std::make_pair(hinge, neighbor)] + heuristic(neighbor, dst_id);
                priority_queue.push_back(std::make_pair(neighbor, new_heuristic));
                std::push_heap(priority_queue.begin(), priority_queue.end(), cmp);
            }
        }
    }

    //proof - formalized and improved while at TigerGraph - the fastest MPP usssp (unsigned single source fastest path) by using pruning conditions for non-negative edges (works especially well for sparse graph (says, Google Maps - when A, B are deterministic - and all pairs are required - think of an all-pair a_star) - because the complexity then is O(M * edge_count(A, B)) practically much lower - wheras dijkstra is a fixed O(N * log(n) + M)
    //assume there is the shortest path between A and B
    //WLOG, say that A -> C -> D -> E -> B
    //for the next iteration, it's guaranteed to release the responsibily of C, such that every reachables through C is reachable via the remaining nodes - until there is A - B left - which is a direct path - a minimum iteration of max(min_edge_size(A, B)) is required, for A, B are two random points c node_set
    
    void bellman_ford(){

    }

    auto bfs(size_t src_id, const std::unordered_map<size_t, std::vector<size_t>>& edges) -> std::unordered_map<size_t, size_t>{

        std::unordered_map<size_t, size_t> hop_map{};
        std::vector<size_t> queue{src_id};
        size_t hop{};
        hop_map[src_id] = hop;

        while (!queue.empty()){
            std::vector<size_t> tmp{};

            for (size_t src: queue){
                for (size_t dst: edges[src]){
                    if (!hop_map.contains(dst)){
                        tmp.push_back(dst);
                        hop_map[dst] = hop;
                    }
                }
            }

            queue = std::move(tmp);
            hop += 1;
        }

        return hop_map;
    }

    void dfs(size_t src_id, size_t hop, const std::unordered_map<size_t, std::vector<size_t>>& edges, std::unordered_map<size_t, size_t>& hop_map){

        if (hop_map.contains(src_id)){
            return;
        }
        
        hop_map[src_id] = hop;

        for (size_t i = 0u; i < edges[src_id].size(); ++i){
            dfs(edges[src_id][i], hop + 1, edges, hop_map);
        }
    }

    auto dfs(size_t src_id, const std::unordered_map<size_t, std::vector<size_t>>& edges) -> std::unordered_map<size_t, size_t>{

        auto rs = std::unordered_map<size_t, size_t>{};
        dfs(src_id, 0u, edges, rs);

        return rs;
    }
} 

#endif