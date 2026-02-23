from collections import defaultdict, deque
from itertools import combinations
from scipy.spatial import KDTree
from itertools import combinations
from scipy.spatial import Delaunay, ConvexHull
from pathlib import Path
import random
import numpy as np
import plotly.graph_objects as go
import time
import math
import copy
import heapq
import concurrent.futures
import csv
import os
import pickle
import warnings
import argparse


# Construct trees
class ConstructTreeProblem:
    def __init__(self, depots_nodes_position, depots, st_pairs):
        self.positions = depots_nodes_position
        self.depots = depots
        self.st_pairs = st_pairs
        self.edge_distance = self.calculate_edge_distance()

    def calculate_edge_distance(self):
        """
        Calculate edge distance (between depot and source) based on positions
        """
        edge_distance = defaultdict(dict)
        for depot in self.depots:
            for source, _ in self.st_pairs:
                distance = euclidean_distance(self.positions[depot], self.positions[source]) # distance between depot and source
                edge_distance[(depot, source)] = distance
        return edge_distance
    
    def merge_dicts_with_list_values(self, d1, d2):
        merged = defaultdict(list)
        for d in (d1, d2):
            for k, v in d.items():
                merged[k].extend(v)
        
        return dict(merged)
    
    def add_same_location_depots(self, same_position_depot_list, target_sources_edges):
        # Depots located in the same location have the same chance of being updated,
        # so we add them to the target_sources_edges here.
        depot2group = {}
        for group in same_position_depot_list:
            for d in group:
                depot2group[d] = set(group)

        for key, values in target_sources_edges.items():
            values = set(values)
            expanded = set(values)

            for d in values:
                if d in depot2group:
                    expanded |= depot2group[d]

            target_sources_edges[key] = expanded

        return target_sources_edges

    # Construct trees
    def construct_mst_with_super_depot(self, constant_MST, worst_instance_greedy = False):
        """
        Construct MST by merging depots into a super node, computing MST, and then un-contracting depots.
        (This part can be understood as: convert a 2D problem to 3D, solve the MST, and then convert it back to 2D.)
        Using a super_depot is helpful, because we don't want to consider the edges between depots.
        """
        super_depot = "r_super"
        initial_depot_dic = {}  # Record the initial depot of each source.
        depot_to_source_edges = []
        voronoi_delaunay_edges = []
        nodes = [source for source, _ in self.st_pairs]
        
        self.step_timing_log = []

        t1 = time.perf_counter()

        # Step 1: Get the min_cost edge between depot and source, and store them in the depot_to_source_edges
        for source, _ in self.st_pairs:
            min_cost = float('inf')
            for depot in self.depots:
                depot_source_distance = self.edge_distance[(depot, source)]
                if depot_source_distance < min_cost:    # If two (depot, source) are same, then we use the front depot.
                    min_cost = depot_source_distance
                    initial_depot = depot   # Record the initial depot of this source, we will assign a super_depot to this source soon.
            # After checking all the edges between this source and the depots, we found the min_cost edge.
            # Record r_super-to-sources edge, containing edge weight, super_depot and source
            depot_to_source_edges.append((min_cost, super_depot, source))
            initial_depot_dic[source] = initial_depot
        
        # Calculate the edges(euclidean_distance) between nodes, and add them to the new_edges (the edge list).
        # voronoi_delaunay helps us: for a node, we only need to consider its "nearest neighbor". We don't need to consider edges connected to distant nodes.
        # Our nodes are fixed and unchanging, so the edges obtained by delaunay are fixed. If any nodes change (more or fewer), we need to rebuild delaunay to obtain the edges.
        
        if worst_instance_greedy:
            target_sources_edges = {}
            pts, adjS, _, _ = voronoi_delaunay_sources(nodes, worst_instance_greedy)            
            for s, t in st_pairs:
                target_sources_edges[t] = sorted(set(adjS.get(s, set())))

        else:
            ############### For using Prim TC ################
            target_fill_in_sourceDelaunay = True
            source_fill_in_targetDelaunay = False
            ##################################################
            same_position_depot_list1 = []
            target_sources_edges1 = {}
            if target_fill_in_sourceDelaunay:
                
                tri, pts, adjS, voronoi_delaunay_edges, same_position_depot_list1 = voronoi_delaunay_sources(nodes)            
                for s, t in st_pairs:
                    Nt = natural_neighbors_of_t(tri, pts, nodes, depots_nodes_position[t])
                    if s in Nt:
                        chosen = set(adjS.get(s, set())) - {s}
                    else:
                        chosen = Nt
                    target_sources_edges1[t] = sorted(chosen)
            
            same_position_depot_list2 = []
            target_sources_edges2 = {}
            if source_fill_in_targetDelaunay:
                nodes = [target for _,target in self.st_pairs]
                tri, pts, adjS, voronoi_delaunay_edges, same_position_depot_list2 = voronoi_delaunay_sources(nodes)
                source_targets_edges = {}
                for s, t in st_pairs:
                    Nt = natural_neighbors_of_t(tri, pts, nodes, depots_nodes_position[s])
                    if t in Nt:
                        chosen = set(adjS.get(t, set())) - {t}
                    else:
                        chosen = Nt
                    source_targets_edges[s] = sorted(chosen)

                d_new = defaultdict(list, {n:[] for n in nodes})
                for k, vals in source_targets_edges.items():
                    for v in vals:
                        d_new[v].append(k)
                target_sources_edges2 = dict(d_new)
            
            target_sources_edges = self.merge_dicts_with_list_values(target_sources_edges1, target_sources_edges2)
            same_position_depot_list1.extend(same_position_depot_list2)
            if same_position_depot_list1:
                target_sources_edges = self.add_same_location_depots(same_position_depot_list1, target_sources_edges)


        nodes.append(super_depot)
        
        t2 = time.perf_counter()

        # Tree combination use prim algorithm to calculate the MST and Construct Tree.
        # (ps. super_depot --> nearest source1 --> source1's target --> nearest source2 --> source2's target --> ...--> last source --> last source's target)
        # Step 2: Compute MST using Prim Algorithm. In this method, we caluculate the MST including source to target edge
        if constant_MST[0]:
            mst_edges = prim_mst_ss_st(depot_to_source_edges, target_sources_edges, adjS, constant_MST[1])
        else:
            mst_edges = prim_mst(depot_to_source_edges, target_sources_edges)
        
        t3 = time.perf_counter()

        # Step 3: Un-contract super_depot and restore original depots
        mst_result_trees = []
        for weight, u, v in mst_edges:
            if u == super_depot:
                initial_depot = initial_depot_dic[v]
                mst_result_trees.append((initial_depot, v, weight))
            else:
                mst_result_trees.append((u, v, weight))
        
        t4 = time.perf_counter()

        # when we use prim, we caluculate the MST including source to target edge, so mst_result_trees is the final_trees.
        self.step_timing_log.append({
            "step1": t2-t1,
            "step2": t3-t2,
            "step3": t4-t3,
        })

        return mst_result_trees, []      # mst_result_trees: mst among source and target

        
    def tree_bfs(self, graph, start):
        queue = deque([start])
        visited = set()
        visited.add(start)
        tree_nodes = []     
        total_distance = 0
        while queue:
            node = queue.popleft()
            tree_nodes.append(node)
            for neighbor, weight in graph[node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        total_distance += weight
                        queue.append(neighbor)                               
                        
        return tree_nodes, total_distance    
    
    # def tree_dfs(self, graph, start):
    #     tree_nodes = []    
    #     visited = set()
    #     def dfs(node):
    #         visited.add(node)  
    #         tree_nodes.append(node)
    #         distance = 0
    #         for neighbor, weight in graph[node]:
    #             if neighbor not in visited:
    #                 distance += weight + dfs(neighbor) # current edge weight + counted edge weight
    #         return distance
    #     total_distance = dfs(start)
    #     return tree_nodes, total_distance
       
    def tree_dfs(self, graph, start):
        tree_nodes = []
        visited = set([start])
        total_distance = 0

        # stack entries: (node, iterator over neighbors)
        stack = [(start, iter(graph.get(start, [])))]
        tree_nodes.append(start)

        while stack:
            node, it = stack[-1]
            try:
                neighbor, weight = next(it)
            except StopIteration:
                stack.pop()
                continue

            if neighbor in visited:
                continue

            visited.add(neighbor)
            tree_nodes.append(neighbor)
            total_distance += weight

            stack.append((neighbor, iter(graph.get(neighbor, []))))

        return tree_nodes, total_distance
    
    def build_trees_from_prim_result(self, mst_result_trees):
        graph = defaultdict(list)
        for u, v, weight in mst_result_trees:
            graph[u].append((v,weight))     # our prim mst_result_trees: depot --> souce1-->target1 --> souce2-->target2 --> souce N-->target N

        trees = {depot: self.tree_dfs(graph, depot) for depot in depots}
        return trees

    def build_trees(self, mst_result_trees, source_target_edge):
        """
        Based on the mst and s_t pair, for each tree, search all the nodes and calculate the weight.
        Input: MSTrees and s_t pair
        Output: Trees e.g. {depot: ({all the nodes}, weight of tree)}
        """
        # 1. BFS traverses the graph and assigns the nodes of each MSTree.
        # Record the tree_nodes and Calculate the total_distance of MSTree
        graph = defaultdict(list)
        for u, v, weight in mst_result_trees:
            graph[u].append((v,weight))
            graph[v].append((u,weight))

        trees = {depot: self.tree_bfs(graph, depot) for depot in depots}
        
        # 2. Add the S-T pair on MSTree
        # Record the target_nodes and the distance of S-T pairs, then add them to the each of MSTree.
        source_target_dict = defaultdict(list)
        for source, target, weight in source_target_edge:
            source_target_dict[source].append((target, weight))
            
        entire_trees = {}

        # running time: O(number of nodes * number of s_t pair)
        for depot, (tree_nodes, total_distance) in trees.items():   # for loop number of trees (depots)
            target_nodes = set()
            s_t_distance = 0
            for tree_node in tree_nodes:                            # for loop number of nodes in each tree
                for target, weight in source_target_dict.get(tree_node, []):
                    target_nodes.add(target)
                    s_t_distance += weight
                        
            entire_trees[depot] = ((tree_nodes.union(target_nodes)), (total_distance + s_t_distance))

        total_weight_from_edge = sum(weight for _, _, weight in source_target_edge) + sum(weight for _, _, weight in mst_result_trees)
        total_weight_from_graph = sum(total_distance for key, (tree_nodes, total_distance) in entire_trees.items())
        if abs(total_weight_from_edge - total_weight_from_graph)>1:
            raise Exception(f"Something wrong with total_weight of trees. Please have a check")

        return entire_trees

     
    # Function to traverse and find all nodes connected to a given depot
    def get_mst_total_distance(self, depot, trees):
        visited = set()
        stack = [depot]
        total_distance = 0

        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)

            for neighbor, weight in trees[node]:
                if neighbor not in visited:
                    total_distance += weight
                    stack.append(neighbor)

        return total_distance
    
    # plotly
    def plot_trees_plotly(self, mst_result, plot_range):
        fig = go.Figure()

        # Plot nodes
        for node, (x, y) in self.positions.items():
            if node not in self.depots:
                is_source = node in [s for (s, _) in self.st_pairs]
                fig.add_trace(go.Scatter(
                    x=[x], y=[y],
                    mode='markers+text',
                    marker=dict(
                        size=20,
                        color='#64afdc' if is_source else 'white',
                        line=dict(color='black' if is_source else '#64afdc', width=1)
                    ),
                    text=[str(node)],
                    textposition='middle center',
                    textfont=dict(size=8, color='black'),
                    showlegend=False
                ))

        # Plot depots
        for depot in self.depots:
            x, y = self.positions[depot]
            fig.add_trace(go.Scatter(
                x=[x], y=[y],
                mode='markers',
                marker=dict(size=16, color='lightgreen', line=dict(color='green', width=2), symbol='triangle-up'),
                name='Depot'
            ))

        # Draw MST edges
        for start, end, _ in mst_result:
            x0, y0 = self.positions[start]
            x1, y1 = self.positions[end]
            fig.add_trace(go.Scatter(
                x=[x0, x1], y=[y0, y1],
                mode='lines',
                line=dict(color='green', width=1),
                opacity=0.7,
                showlegend=False
            ))

        # Draw source-target pairs
        for source, target in self.st_pairs:
            x0, y0 = self.positions[source]
            x1, y1 = self.positions[target]
            fig.add_trace(go.Scatter(
                x=[x0, x1], y=[y0, y1],
                mode='lines',
                line=dict(color='orange', width=1),
                showlegend=False
            ))

        # Layout settings
        fig.update_layout(
            width=800,
            height=800,
            title="Tree & ST Pair Plot",
            xaxis=dict(
                tickmode='linear',
                tick0=0,
                dtick=1,
                # range=[-0.5, plot_range],
                range=[174, 175], 
                mirror=True,
                showgrid=True,
                zeroline=False
            ),
            yaxis=dict(
                tickmode='linear',
                tick0=0,
                dtick=1,
                # range=[plot_range, -0.5],  # Invert y-axis
                range=[45.5, 46.5], 
                mirror=True,
                showgrid=True,
                zeroline=False
            ),
            showlegend=False
        )

        # Save to file
        save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"plot_trees_plotly.html")
        fig.write_html(save_path)

# Primal_Dual Algorithm
class PrimalDualAlgorithm:
    def __init__(self, levels, positions, trees_structure, weights):
        self.levels = levels  # Levels with depots, targets, and speeds
        self.weights = weights # Weights of trees
        self.positions = positions  # 2D coordinates of all depots and targets
        self.trees_structure = trees_structure # The trees formed by ConstructTreeProblem
        self.active_components = defaultdict(list)  # Active components per depot
        self.dual_variables = defaultdict(lambda: defaultdict(float))  # Dual variables Y_i(S)
        self.internal_variables = defaultdict(lambda: {"p": defaultdict(float), "Y_bar": defaultdict(float), "Bound": defaultdict(float), "Pi": defaultdict(float)})  # Internal variables
        self.edge_costs, self.tree_distances = self.calculate_edge_costs()  # Between trees.
        self.steps = []  # Store steps dynamically
        self.new_edges = defaultdict(lambda: defaultdict(list))
        self.previously_frozen_nodes = defaultdict(lambda: defaultdict(list))
    
    def build_kdtree(self, nodes):
        coords = [self.positions[n] for n in nodes]
        return KDTree(coords), nodes

    def compute_min_dist(self, kdtree_tuple, other_tree_nodes):
        # min_dist is the minimum distance between two trees
        # nodeA and nodeB are the nodes connecting these two trees
        treeA, nodesA = kdtree_tuple
        other_nodes_list = list(other_tree_nodes)
        coordsB = [self.positions[n] for n in other_nodes_list]
        dists, idxs = treeA.query(coordsB)
        min_idx_in_B = int(np.argmin(dists))
        min_dist     = float(dists[min_idx_in_B])
        nodeB = other_nodes_list[min_idx_in_B]
        nodeA = nodesA[idxs[min_idx_in_B]]
        return min_dist, nodeA, nodeB

    def parallel_compute_tree_distances(self, max_workers):
        # 1) build KD-Tree for each tree
        kdtrees = {}
        for tree_id, nodes in self.trees_structure.items():
            kdtrees[tree_id] = self.build_kdtree(list(nodes))

        # 2) generate all tree pairs, e.g.(treeA, treeB)
        tree_pairs = list(combinations(self.trees_structure.keys(), 2))
        results = {}

        def worker(pair):
            t1, t2 = pair
            dist, nodeA, nodeB = self.compute_min_dist(kdtrees[t1], self.trees_structure[t2])
            return pair, dist, nodeA, nodeB

        # 3) Parallel computing
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(worker, pair) for pair in tree_pairs]
            for fut in concurrent.futures.as_completed(futures):
                pair, dist, nodeA, nodeB = fut.result()
                results[pair] = {
                    "distance": dist,
                    "node_from_A": nodeA,
                    "node_from_B": nodeB,
                }

        return results
    
    def calculate_edge_costs(self):
        """
        Calculate edge costs (travel times) based on *the minimum distance between two trees* and the speed of the level.
        Minimum distance between two trees
        """
        edge_costs = defaultdict(dict)
        
        tree_distances = self.parallel_compute_tree_distances(max_workers=4)

        for level, data in self.levels.items():
            speed = data["speed"]
            depots = data["depots"]
            targets = data["targets"]
            
            # We do not need the edge_costs between depots (because same level's depots are inactive.), so we caculate the edge_costs of depot-target and target-target below.
            # from depot to target
            for depot in depots:
                for target in targets:
                    if tree_distances.get((depot, target)) is None:        
                        raise Exception(f"({depot}, {target}) not in tree_distances. Please have a check")
                    distance = tree_distances[(depot,target)]['distance']
                    cost = distance / speed
                    edge_costs[level][(depot, target)] = cost

            # from target to target
            for target in targets:
                for other_target in targets:
                    if target != other_target:
                        if (other_target, target) not in edge_costs[level]:  # Avoid store duplicate edge
                            if tree_distances.get((target, other_target)) is None:        
                                raise Exception(f"({target}, {other_target}) not in tree_distances. Please have a check")
                            distance = tree_distances[(target, other_target)]['distance']
                            cost = distance / speed
                            edge_costs[level][(target, other_target)] = cost

        return edge_costs,tree_distances


    def initialize_components(self):
        """
        Initialize components and dual variables for all levels.
        """
        last_level = max(self.levels.keys()) # The last level

        for level, data in self.levels.items():
            depots = data["depots"]
            targets = data["targets"]

            for depot in depots:
                self.active_components[level].append(((depot,), "inactive"))  # Depots: Start as "inactive"
                self.dual_variables[level][(depot,)] = 0.0  # Depot dual variable is always 0
                self.internal_variables[level]["p"][depot] = 0.0  # Depot's initially accumulated dual value is 0

            for target in targets:
                self.active_components[level].append(((target,), "active"))  # Targets: Start as "active"
                self.dual_variables[level][(target,)] = 0.0
                self.internal_variables[level]["p"][target] = 0.0  # Target's initially accumulated dual value is 0
                self.internal_variables[level]["Y_bar"][(target,)] = 0.0
                self.internal_variables[level]["Bound"][(target,)] = 0.0

            
    def compute_max_dual_growth(self, C7_time, C8_time):
        """
        Compute the maximum allowable dual variable growth (\Delta_1) for all active components.
        Includes checking for both Constraint (7) and Constraint (8).
        """
        delta1 = float('inf')
        delta2 = float('inf')
        zero_dual_value = False

        # We don't need to consider the last level, because all the vertices in the last level are inactive at the begining.
        levels_list = list(self.levels.keys())      # self.levels.keys(): dict_keys([1, 2, 3])
        levels_list_non_last = levels_list[:-1]     # Removes the last level 

        constrain_timing_log = []

        for i, level in enumerate(levels_list_non_last):
            data = self.levels[level]
            speed = data["speed"]

            # Get speed of next level.
            next_level_speed = self.levels[levels_list[i + 1]]["speed"]
            
            t7 = time.perf_counter()
          
            for (u, v), cost in self.edge_costs[level].items():
                # Check whether u and v comes from the same component. If they come same component, we don't need to calculate the dual value between them, that is to say they had already connected.
                node_to_component = {}
                for comp, state in self.active_components[level]:
                    nodes = comp if isinstance(comp, (list, tuple)) else [comp]
                    for node in nodes:
                        node_to_component[node] = (comp, state)
                u_component, u_state = node_to_component.get(u, (None, None))
                v_component, v_state = node_to_component.get(v, (None, None))

                if u_component == v_component:
                    continue

                active_c1 = 1 if u_state == "active" else 0
                active_c2 = 1 if v_state == "active" else 0
                
                if (u_state == "active") or ( v_state == "active"):
                    # get min dual value among target-target and depot-target pairs
                    # Calculate maximum allowable growth for edge (u, v) 
                    current_p_sum = self.internal_variables[level]["p"][u] + self.internal_variables[level]["p"][v]
                    max_growth = (cost - current_p_sum) / (active_c1 + active_c2)
                    delta1 = min(delta1, max_growth)
                
                    if delta1 == 0: 
                        zero_dual_value = True
                        break
            if zero_dual_value == True:
                break
            

            t8 = time.perf_counter()
            
            # Constraint (8): Component-related growth
            for (component, state) in self.active_components[level]:   
                if state == "active":
                    active_descendants = self.check_active_descendants(level, component)
                    # Only consider components with no active descendants. 
                    # If m_C > 0, both level i and level i+1 can increase dual value simultaneously, no problem at all.
                    # If m_C = 0, the max dual level i can add is (Bound_C - Y_bar_C), because there is a limit dual_value in level i+1, level i cannot beyond it. If beyond, then it violate Constraint (8).
                    if active_descendants == 0:
                        Y_bar_C = self.internal_variables[level]["Y_bar"][component]
                        Bound_C = self.internal_variables[level]["Bound"][component] 
                        
                        Weight_C = 0
                        for ele in component:
                            Weight_C = Weight_C + self.weights[ele]
                        Pi_C = (Weight_C/next_level_speed) - (Weight_C/speed)
                        
                        self.internal_variables[level]["Pi"][component] = Pi_C  # Store Pi, when the component arrives the frozon limitation

                        max_growth = Bound_C - Y_bar_C + Pi_C
                        delta2 = min(delta2, max_growth)
                
            t9 = time.perf_counter()
            
            # Record timing info
            constrain_timing_log.append({
                "level": level,
                "C7": t8 - t7,
                "C8": t9 - t8               
            })

        C7 = sum(step.get("C7") for step in constrain_timing_log)
        C7_time = C7 if C7 > 0 else 0

        C8 = sum(step.get("C8") for step in constrain_timing_log)
        C8_time = C8 if C8 > 0 else 0

        delta_1_2 = 1 if delta1 <= delta2 else 2
        
        return min(delta1, delta2), delta_1_2, C7_time, C8_time

    def grow_dual_variables(self, delta):
        """
        Increase dual variables for all active components by the given \Delta.
        Apply the same \Delta to every active node.
        """
        # We don't need to consider the last level, because all the vertices in the last level are inactive at the begining.
        levels_list = list(self.levels.keys())      # self.levels.keys(): dict_keys([1, 2, 3])
        levels_list_non_last = levels_list[:-1]     # Removes the last level 

        for _, level in enumerate(levels_list_non_last):
            for (component, state) in self.active_components[level]:
                if state == "active":
                    if len(component) >= 1:
                        for node in component:
                            self.internal_variables[level]["p"][node] += delta      # accumulate all the dual_value for a node
                    self.dual_variables[level][component] += delta
                    active_descendants = self.check_active_descendants(level, component)
                    self.internal_variables[level]["Y_bar"][component] += delta
                    self.internal_variables[level]["Bound"][component] += active_descendants * delta
 
    def update_components(self, delta_1_2, step_number):
        """
        Check constraints and update components (merge, freeze, deactivate).
        """
        # We don't need to consider the last level, because all the vertices in the last level are inactive at the begining.
        levels_list = list(self.levels.keys())      # self.levels.keys(): dict_keys([1, 2, 3])
        levels_list_non_last = levels_list[:-1]     # Removes the last level 

        for _, level in enumerate(levels_list_non_last):
            if any(comp[1] == "active" for comps in self.active_components.values() for comp in comps):  # During "merge_components/deactivate_descendants" or "frozen_descendant", it is possible that all the descendants would be deactivated. We should not update the inactivated or frozen components.
                # If multiple constraints in both (7) and (8) become tight simultaneously during iteration t, only one constraint either in (7) or (8) is chosen.
                # If multiple constraints in (7) become tight simultaneously during iteration t, all tight component will be processed, same for (8).
               
                if delta_1_2 == 1:    
                    for (u, v), cost in self.edge_costs[level].items():
                        node_to_component = {}
                        for comp, state in self.active_components[level]:
                            nodes = comp if isinstance(comp, (list, tuple)) else [comp]
                            for node in nodes:
                                node_to_component[node] = (comp, state)
                        u_component, u_state = node_to_component.get(u, (None, None))
                        v_component, v_state = node_to_component.get(v, (None, None))

                        if u_component == v_component:
                            continue
                        
                        # Constraint (7) tight: Merge components
                        if (u_state == "active") or ( v_state == "active"): # Note: Please be careful. Without this judge, two inactive components could be connected.
                            current_dual_sum = self.internal_variables[level]["p"][u] + self.internal_variables[level]["p"][v]
                            if current_dual_sum >= cost:
                                if current_dual_sum > cost + 0.00000000000001: # control minimum difference !!!
                                    raise Exception("This is an 'current_dual_sum > cost' error. Please have a check")
                                if (u_component + v_component) not in self.active_components[level] and (v_component + u_component) not in self.active_components[level]:
                                    self.merge_components(level, u_component, v_component)
                                    self.new_edges[level][step_number].append((u, v))
                                else:
                                    raise Exception("(u_component + v_component) or (v_component + u_component) already in self.active_components[level]. Please have a check")

                # "delta_1_2 == 2": delta2 < delta1, constraint 8 gives smaller delta
                if delta_1_2 == 2:
                    # Check constraint (8) and freeze components if tight  
                    for (component, state) in self.active_components[level]:
                        if state == "active":
                            active_descendants = self.check_active_descendants(level, component)
                            if active_descendants == 0:  # Only proceed if no active descendants
                                bound = self.internal_variables[level]["Bound"][component]
                                Y_bar = self.internal_variables[level]["Y_bar"][component] # Here Y_bar had added duel_value in grow_dual_variables()
                                Pi = self.internal_variables[level]["Pi"][component]
                                if Y_bar >= bound + Pi: # "Y_bar < bound + Pi" means that haven't been bounded. Y_bar still have space to grow.
                                    self.freeze_component(level, component)
                                    if Y_bar > bound + Pi:
                                        raise Exception(f"Found Y_bar > bound + Pi; Please have a check. level: {level}, (component, state): ({component}, {state}).")


    def check_active_descendants(self, level, component):
        """
        Check the components in level (i+1), whether they are active and whether they are a subset of component
        :param level: The current level (i).
        :param component: The component to inspect.
        :return: A list of active subsets in the next level (i+1).
        # Running time: O(len(active_components[next_level])* min(len(component),len(active_components[next_level])))
        """
        # Ensure the component is iterable
        if isinstance(component, int):
            component = [component]

        active_subsets = []
        next_level = level + 1  # Move to level (i+1)

        # Convert the component to a set once for faster comparison
        component_set = set(component)

        if next_level <= len(self.levels):  # Ensure the next level exists
            for comp, state in self.active_components[next_level]:
                # Directly check if comp is a subset of component_set and is active
                if state == "active":
                    if set(comp).issubset(component_set):
                        active_subsets.append(comp)

        return len(active_subsets)
    

    def merge_components(self, level, u_component, v_component):
        """
        Merge two components (u_component, v_component) for the given level, and initial its dual value (0)
        Check if the merged component contains a depot. If so, mark it as inactive.
        """
        # Combine nodes from both components
        # new_component = tuple(set(u_component) | set(v_component))  # Use tuple for hashable keys
        new_component = u_component + v_component   # By this way, we can get the fix order of new component

        # Remove old components (u_component and v_component) from active_components
        self.active_components[level] = [
            (component, state)
            for component, state in self.active_components[level]
            if set(component) != set(u_component) and set(component) != set(v_component)
        ]

        # Check if any depot is in the merged nodes
        contains_depot = any(node in self.levels[level]["depots"] for node in new_component)

        # Update component state
        component_state = "inactive" if contains_depot else "active"

        # Add the new merged component
        self.active_components[level].append((new_component, component_state))

        # Initial its dual value (0)
        self.dual_variables[level][new_component] = 0


        # If the merged component is inactive, deactivate its descendants
        if component_state == "inactive":
            self.deactivate_descendants(level, new_component)

        # print(f"Level {level}: Merged components {u_component} and {v_component}.We are getting {new_component} New state: {component_state}.")
        
        # Update internal variables for the merged component
        Y_bar_u = self.internal_variables[level]["Y_bar"][u_component]
        Y_bar_v = self.internal_variables[level]["Y_bar"][v_component]
        Bound_u = self.internal_variables[level]["Bound"][u_component]
        Bound_v = self.internal_variables[level]["Bound"][v_component]

        # Combine the Y_bar and Bound values
        self.internal_variables[level]["Y_bar"][new_component] = Y_bar_u + Y_bar_v
        self.internal_variables[level]["Bound"][new_component] = Bound_u + Bound_v


    def deactivate_descendants(self, level, merged_nodes):
        """
        Deactivate all descendants of the merged component if it becomes inactive.
        """
        # Iterate over all layers after the current level layer       
        for next_level in range(level + 1, len(self.levels) + 1):
            for idx, (component, state) in enumerate(self.active_components[next_level]):
                # Check if the component is a subset of merged_nodes and is active
                if set(component).issubset(set(merged_nodes)) and state == "active":
                    self.active_components[next_level][idx] = (component, "inactive")
                    # print(f"Level {next_level}: Component {component} deactivated as part of merged component.")


    def freeze_component(self, level, node):
        """
        Freeze a component when constraint (8) becomes tight.
        A component is frozen if it stopped growing due to the constraint in (8), 
        i.e., each descendant of this component is not active as it contains targets connected to some higher level depots.
        """
        for idx, (n, state) in enumerate(self.active_components[level]):      
            if n == node:
                self.active_components[level][idx] = (n, "frozen")
        # print(f"Level {level}: Component {node} frozen due to tight constraint(8).")

        # frozen_descendant
        for next_level in range(level + 1, len(self.levels) + 1):       
            for idx, (component, state) in enumerate(self.active_components[next_level]):
                # Check if the component is a subset of merged_nodes and is active
                if set(component).issubset(set(node)) and state == "active":
                    self.active_components[next_level][idx] = (component, "inactive")
                    # print(f"Level {next_level}: Component {component} deactivated as part of merged component.")
    
    
    def prune_forests(self, levels, final_components, edges, historical_frozen_component):
        pruned_forests = {}  # Store pruned forests at each level
        for i in range(1, len(levels) + 1):  # Iterate over levels
            Gi = final_components.get(i, [])  # Get the forest at level i

            if i == len(levels): # For the last level, there is no frozen component.
                previously_level_nodes = [v for level in range(1, len(levels)) for component, status in pruned_forests[level] for v in component]
                last_level_v = [v for comp,_ in Gi for v in comp] 
                if any(v in previously_level_nodes for v in last_level_v):
                    raise Exception(f"Node(s) at the last level also appear at the previous level. Please have a check. last_level_v: {last_level_v}, previously_level_nodes: {previously_level_nodes}. ")
                # Store pruned forest for the last level
                pruned_forests[i] = Gi
                # print(Gi)
                continue
            
            frozen_components = [comp for comp, status in final_components[i] if status == "frozen"]
            depots = set(levels[i]["depots"])  # Get depots at this level
            
            # Step 1: Remove frozen components that do not contain depots
            Gi_with_depot = []
            removed_no_depot = set()
            for comp, status in Gi:
                if status == 'frozen' and not any(node in depots for node in comp):
                    removed_no_depot.update(comp)
                else:
                    Gi_with_depot.append((comp, status))
            Gi = Gi_with_depot

            # Step 2: Compute degrees of each component
            degree_count = defaultdict(int)
            
            if i in historical_frozen_component:   
                for frozen_component in historical_frozen_component[i]:
                    # check whether all nodes in frozen_component are fully contained in a tuple of Gi
                    if any(set(frozen_component).issubset(set(comp)) for comp,_ in Gi):
                        # counting the edges connecting with this frozen_component
                        for edge in edges[i]:
                            u, v = edge
                            # Check if either u or v is in this frozen_component
                            if (u in frozen_component and v not in frozen_component) or (u not in frozen_component and v in frozen_component):
                                degree_count[frozen_component] += 1

            # Step 3: Remove maximal pendent-frozen subgraphs
            # get the maximal pendent-frozen subgraphs
            max_components = {}
            for key in degree_count:
                is_maximal = True
                for other_key in degree_count:
                    if degree_count[key] == 1 and degree_count[other_key] == 1:
                        if key != other_key and set(key).issubset(set(other_key)):
                            is_maximal = False
                            break
                if is_maximal:
                    max_components[key] = degree_count[key]

            # all the elements in the maximal pendent-frozen subgraphs
            elements_to_remove = set()
            removed_MPFS = []
            for component in max_components:
                if max_components[component] == 1:
                    removed_MPFS.append(component)
                    elements_to_remove.update(component)

            # remove all there elements
            Gi = [(tuple(ele for ele in tpl if ele not in elements_to_remove), status) for tpl, status in Gi]

            # Store pruned forest for level i
            pruned_forests[i] = Gi

            # Step 4: Prepare next-level forests by adding removed frozen subgraphs
            if i < len(levels):  # If not the last level
                next_level = i + 1
                # Add both MPFS and removed_no_depot components
                pass_to_next_level_vertices = removed_no_depot.union(elements_to_remove)
                filtered_next_level = []
                for comp, status in final_components.get(next_level, []):
                    # Keep only the nodes from the previous level’s pass-down set
                    new_comp = tuple(v for v in comp if v in pass_to_next_level_vertices)
                    if new_comp:  # Only keep non-empty components
                        filtered_next_level.append((new_comp, status))

                final_components[next_level] = filtered_next_level

        # Check if all the components are inactive, otherwise raise Exception 
        if any([value != 'inactive' for level in pruned_forests for _,value in pruned_forests[level]]):
            raise Exception("The status of some components are not 'inactive'. Please have a check.")

        return pruned_forests
    

    def get_set_from_dic(self, dic):
        new_dic = {}
        for key, sub_dict in dic.items():
            unique_values = set()
            for values in sub_dict.values():
                unique_values.update(values)
            new_dic[key] = unique_values
        return new_dic
    

    def capture_step(self, step_number, delta):
        """
        Capture the state of the system at the current step.
        """
        for level, _ in self.levels.items():
            active = []
            frozen = []
            inactive = []
            for component, state in self.active_components[level]:
                if state == "active":
                    active.append(list(component))
                elif state == "frozen":
                    frozen.append(list(component))
                    self.previously_frozen_nodes[level][step_number].append(component) # record the process of all of the frozen nodes
                elif state == "inactive":
                    inactive.append(list(component))
            
            speed = self.levels[level]['speed']

            self.steps.append({
                "step": step_number,
                "level": level,
                "active": active,
                "frozen": frozen,
                "inactive": inactive,
                "dual_value": delta,
                "speed": speed
            })        


    def run_algorithm(self):
        """
        Run the primal-dual algorithm until all components are inactive or frozen.
        """
        self.initialize_components()
        step_number = 1

        self.step_timing_log = []
        

        while any(comp[1] == "active" for comps in self.active_components.values() for comp in comps):
            C7_time = 0
            C8_time = 0
            t1 = time.perf_counter()
            delta, delta_1_2, C7_time, C8_time  = self.compute_max_dual_growth(C7_time, C8_time)

            t2 = time.perf_counter()
            # If delta == 0, we don't need to grow dual value for the node or component.
            if delta > 0:
                self.grow_dual_variables(delta)

            t3 = time.perf_counter()
            self.update_components(delta_1_2, step_number)

            t4 = time.perf_counter()

            self.capture_step(step_number, delta)  # Capture dual_value of current step
            t5 = time.perf_counter()

            # Record timing info
            # self.step_timing_log.append({
            #     "step": step_number,
            #     "compute_max_dual_growth_time": t2 - t1,
            #     "grow_dual_variables_time": t3 - t2,
            #     "update_components_time": t4 - t3,
            #     "total_step_time": t4 - t1,
            #     'capture_step_time': t5 - t4,
            #     'C7_time': C7_time,
            #     'C8_time': C8_time
            # })

            step_number += 1

        # print("Primal-dual algorithm completed.")
        # print("Steps captured:", self.steps)
        # print("components:", dict(self.active_components))
        # print("Final dual variables:", dict(self.dual_variables))


if __name__ == "__main__":
    # const_7_time  = []
    # const_8_time = []
    # compute_max_dual_growth_time = []
    # grow_dual_variables_time = []
    # update_components_time = []
    # total_step_time = []
    # pruning_time = []
    # capture_step_time = []


    # n nodes can generate n/2 s-t pair. (to avoid same s or same t name) make sure every s or t has a unique name, because we will accumulate each node's dual value by "p" (grow_dual_variables). if there are same node name in same level, we will update it mutiple times in one round.
    def generate_st_pairs(nodes, num_pairs):
        node_keys = list(nodes.keys())

        max_pairs = len(node_keys) // 2
        if num_pairs > max_pairs:
            raise ValueError(f"Too many pairs requested. Can only create up to {max_pairs} non-overlapping pairs.")

        st_pairs = []
        for i in range(num_pairs):
            st_pairs.append((node_keys[2 * i], node_keys[2 * i + 1]))

        return st_pairs    

    # Depots and targets should be given in order, otherwise when we calculate the edge cost will raise exception.
    def generate_depots(level_k, K_list):
        depots = []
        for level in range(1, level_k + 1):
            K = K_list[level - 1]  
            depots += [f"{level}.{i+1}" for i in range(K)]  # "1.1", "1.10", "1.11"
        return depots  
    
    def generate_levels(level_k, K_list, base_speed=50.0, speed_decay=5):
        depots = generate_depots(level_k, K_list)  # generate depots
        levels = {}
        depot_speed_dic = {}

        for level in range(1, level_k + 1):
            current_depots = [d for d in depots if d.startswith(f"{level}.")]
            future_depots = [d for d in depots if int(d.split(".")[0]) > level]
            current_speed = max(base_speed - (level - 1) * speed_decay, 0.1)
            for current_depot in current_depots:    # record every depot's speed
                depot_speed_dic[current_depot] = current_speed
            levels[level] = {
                "speed": current_speed,  # make sure the speed > 0.1
                "depots": current_depots,
                "targets": future_depots
            }

        return depots, levels, depot_speed_dic

    def sample_gmm_nodes(n, depots, Z, L, sigma, seed, weights=None):
        rng = np.random.default_rng(seed)

        # 1) mixture weights
        if weights is None:
            weights = np.ones(Z) / Z
        weights = np.array(weights, dtype=float)
        weights = weights / weights.sum()

        # 2) cluster centers uniformly in [0, L]^2
        centers = rng.uniform(0.0, L, size=(Z, 2))

        # 3) covariance = sigma^2 I
        cov = (sigma ** 2) * np.eye(2)

        # 4) sample cluster ids
        ks = rng.choice(Z, size=(n + len(depots)), p=weights) # assign cluster id, according to weights

        # 5) sample points, with rejection to keep inside [0,L]^2
        nodes = np.empty((n, 2), dtype=float)
        depot_nodes = np.empty((len(depots), 2), dtype=float)
        
        for i, k in enumerate(ks):
            while True:
                p = rng.multivariate_normal(mean=centers[k], cov=cov) # get the (x,y), according to cluster id and cov
                if 0.0 <= p[0] <= L and 0.0 <= p[1] <= L:
                    if i < n:
                        nodes[i] = p  # first n nodes are st-pairs
                    else:
                        depot_nodes[i-n] = p  # rest nodes are depots
                    break
        nodes_dict = {
            str(i + 1): (float(nodes[i, 0]), float(nodes[i, 1]))
            for i in range(len(nodes))
        }
         # Generate random depot positions 
        depots_position = { 
            depot: (float(depot_nodes[i, 0]), float(depot_nodes[i, 1]))
            for i, depot in enumerate(depots)}

        return nodes_dict, depots_position, centers, ks

    def plot_graph(depots_position,nodes_position,st_pairs,plot_range):
        fig = go.Figure()

        # Plot nodes
        for node, (x, y) in nodes_position.items():
            is_source = node in [s for (s, _) in st_pairs]
            fig.add_trace(go.Scatter(
                x=[x], y=[y],
                mode='markers+text',
                marker=dict(
                    size=20,
                    color='#64afdc' if is_source else 'white',
                    line=dict(color='black' if is_source else '#64afdc', width=1)
                ),
                text=[str(node)],
                textposition='middle center',
                textfont=dict(size=8, color='black'),
                showlegend=False
            ))

        # Plot depots
        for node, (x, y) in depots_position.items():
            fig.add_trace(go.Scatter(
                x=[x], y=[y],
                mode='markers',
                marker=dict(size=16, color='lightgreen', line=dict(color='green', width=2), symbol='triangle-up'),
                name='Depot'
            ))

        # Draw source-target pairs
        for source, target in st_pairs:
            x0, y0 = nodes_position[source]
            x1, y1 = nodes_position[target]
            fig.add_trace(go.Scatter(
                x=[x0, x1], y=[y0, y1],
                mode='lines',
                line=dict(color='orange', width=1),
                showlegend=False
            ))

        # Layout settings
        fig.update_layout(
            width=800,
            height=800,
            title="depot & ST Pair Plot",
            xaxis=dict(
                tickmode='linear',
                tick0=0,
                dtick=1,
                # range=[-0.5, plot_range],
                range=[174, 175], 
                mirror=True,
                showgrid=True,
                zeroline=False
            ),
            yaxis=dict(
                tickmode='linear',
                tick0=0,
                dtick=1,
                # range=[plot_range, -0.5],  # Invert y-axis
                range=[45.5, 46.5], 
                mirror=True,
                showgrid=True,
                zeroline=False
            ),
            showlegend=False
        )

        # Save to file
        save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"plot_s_t_pair.html")
        fig.write_html(save_path)
    
    # Traverse each tree in order; the nodes in each tree are obtained by previously sorting them according to DFS.
    def get_depot_source_dic(pruned_results,trees_structure):
        source_list = [s for s,_ in st_pairs]
        depot_source_dic = defaultdict(list)
        for level, components in pruned_results.items(): # every level has one/more components
            for component,_ in components:  # each level's components
                source_node = []
                level_depot = None
                for depot in component:  # for the trees in each component
                    if depot.startswith(f'{level}.'):
                        level_depot = depot             # only record the depot in this level 
                    tree = trees_structure.get(depot) # each tree's s_t pair
                    for node in tree:
                        if node in source_list:
                            source_node.append(node)
                if len(source_node)> 0: depot_source_dic[level_depot] = source_node
        return depot_source_dic
    
    def get_depot_source_dic_forGreedyTrees(depot_st_tour_dist_dic):
        source_list = [s for s,_ in st_pairs]
        depot_source_dic = defaultdict(list)
        stTour_distance_dic = {}
        for level, components in pruned_results.items(): # every level has one/more components
            for component,_ in components:  # each level's components
                source_node = []
                level_depot = None
                for depot in component:  # for the trees in each component
                    if depot.startswith(f'{level}.'):
                        level_depot = depot             # only record the depot in this level 
                    if depot in depot_st_tour_dist_dic:
                        ((firstSource,lastTarget),tour_distance) = depot_st_tour_dist_dic[depot]
                        source_node.append(firstSource)
                        stTour_distance_dic[(firstSource,lastTarget)] = tour_distance
                if len(source_node)> 0: depot_source_dic[level_depot] = source_node
        return depot_source_dic,stTour_distance_dic

    def trees_structure_to_depot_source_dic(trees_structure):
        depot_source_dic = defaultdict(list)
        source_list = [s for s,_ in st_pairs]
        for depot, tree in trees_structure.items():
            source_node = []
            for node in tree:
                if node in source_list:
                    source_node.append(node)
            if len(source_node)> 0: depot_source_dic[depot] = source_node
        return depot_source_dic

    def get_depot_edges_dic(pruned_results, new_edges_set):
        edge_list_dic = defaultdict(list)
        for level, components in pruned_results.items():
            for component,_ in components: # e.g. ('1.13', '2.29', '3.7', '3.28', '2.23', '3.4', '3.29', '2.24')
                edge_list = []
                level_depot = None
                for depot in component:
                    if depot.startswith(f'{level}.'):
                        level_depot = depot             # only record the depot in this level 
                        break
                
                for (u,v) in new_edges_set.get(level, []):
                    if (u in component) and (v in component):
                        edge_list.append((u,v))
                
                if len(component) == 0:
                    if len(edge_list) != 0:
                        raise Exception("Number of edge doesn't match with number of component. Please have a check.")
                else:
                    if (len(edge_list)+1) != len(component):    # len(component) might be 0, when we pruned entire component to next level. Otherwise, (len(edge_list)+1) == len(component)
                        raise Exception("Number of edge doesn't match with number of component. Please have a check.")

                if len(edge_list) > 0: edge_list_dic[level_depot] = edge_list
        return edge_list_dic
    
    def orient2d(ax, ay, bx, by, cx, cy):
        return (bx-ax)*(cy-ay) - (by-ay)*(cx-ax)

    def in_circumcircle(px, py, ax, ay, bx, by, cx, cy, eps=1e-12):
        # InCircle test with P as the origin; ABC is in counterclockwise (CCW) order.
        axp, ayp = ax-px, ay-py
        bxp, byp = bx-px, by-py
        cxp, cyp = cx-px, cy-py
        a2 = axp*axp + ayp*ayp
        b2 = bxp*bxp + byp*byp
        c2 = cxp*cxp + cyp*cyp
        det = (
            axp*(byp*c2 - b2*cyp)
        - ayp*(bxp*c2 - b2*cxp)
        + a2 *(bxp*cyp - byp*cxp)
        )
        return det > -eps  # inside/on circumcircle
    
    def nxt(i, n): return (i+1) % n
    def prv(i, n): return (i-1+n) % n
    
    def ordered_hull(points: np.ndarray):
        hull = ConvexHull(points)
        idx  = hull.vertices           # CCW order
        H    = points[idx]
        return H, idx

    # ---------- get two tangent lines ----------
    def external_tangents_to_convex_ccw(H: np.ndarray, t):
        h = len(H)
        if h == 1: return 0, 0
        if h == 2: return 0, 1

        d2 = np.sum((H - np.asarray(t))**2, axis=1)
        i0 = int(np.argmin(d2))

        iR = i0
        while True:
            j = nxt(iR, h)
            ax, ay = t; bx, by = H[iR]; cx, cy = H[j]
            if orient2d(ax, ay, bx, by, cx, cy) < 0:
                iR = j
            else:
                break

        iL = i0
        while True:
            j = prv(iL, h)
            ax, ay = t; bx, by = H[iL]; cx, cy = H[j]
            if orient2d(ax, ay, bx, by, cx, cy) > 0:
                iL = j
            else:
                break

        return iL, iR

    # ---------- get all the edge vertixes between two tangent lines ----------
    def hull_arc_nodes_between_tangents(points: np.ndarray, t, node_ids=None):
        H, hull_idx = ordered_hull(points)
        iL, iR = external_tangents_to_convex_ccw(H, t)

        h = len(H)
        seq_on_H = [iL]
        i = iL
        while i != iR:
            i = nxt(i, h)
            seq_on_H.append(i)

        arc_idx = [int(hull_idx[i]) for i in seq_on_H]
        iL_idx  = int(hull_idx[iL])
        iR_idx  = int(hull_idx[iR])
        
        arc_ids = [node_ids[k] for k in arc_idx]
        iL_id   = node_ids[iL_idx]
        iR_id   = node_ids[iR_idx]

        return arc_ids, iL_id, iR_id
    

    def natural_neighbors_of_t(tri, pts, idx2id, t_xy):
        """
        tri, pts, idx2id: comes from build_delaunay_from_sources
        t_xy: (tx, ty)
        return: set[source_id] t's Delaunay neighbors
        """
        tx, ty = t_xy
        start = tri.find_simplex([t_xy])
        
        # Points outside the triangulation get the value -1.
        # Two possible scenarios
        #   1. t_xy inside one/more triangle's circumcircle --> all the nodes of the triangles + arc_ids
        #   2. t_xy doesn't in any circumcircle --> Just return the arc_ids
        if start == -1:
            arc_ids, _, _ = hull_arc_nodes_between_tangents(
                pts, t_xy, node_ids=idx2id
            )   # all nodes between two side tangents

            # For nodes in arc_ids, check whether their triangular circumcircle including t_xy
            # if including in any triangular circumcircle, update arc_ids
            # otherwise return arc_ids
            
            id2idx = {sid: i for i, sid in enumerate(idx2id)}   # id to index
            # find the index of ids in arc_ids
            arc_idx = np.array([id2idx[sid] for sid in arc_ids], dtype=int) 

            simplices = tri.simplices   # all the tri index
            mask = np.any(np.isin(simplices, arc_idx), axis=1) # any tri includes arc_idx will return True
            cand_tris = np.nonzero(mask)[0] # return True index
            expanded_ids = set(arc_ids) 
            for f in cand_tris:
                a, b, c = simplices[f]
                ax, ay = pts[a]; bx, by = pts[b]; cx, cy = pts[c]
                if orient2d(ax, ay, bx, by, cx, cy) < 0:
                    b, c = c, b
                    bx, by, cx, cy = cx, cy, bx, by
                # if in the circumcircle, add three nodes of the tri to the set
                if in_circumcircle(tx, ty, ax, ay, bx, by, cx, cy):
                    expanded_ids.add(idx2id[a])
                    expanded_ids.add(idx2id[b])
                    expanded_ids.add(idx2id[c])

            return expanded_ids
        
        # If the target t_xy inside any triangle, record this triangle and check this triangle's neighbors.
        # if t_xy inside any neighbor's circumcircle, we also record this neighbor. This means that t_xy distory the property of Delaunay.
        # Thus, we connect t_xy with all these distorted_tri's nodes.

        frontier = [start[0]]           # triangle contains t
        if len(frontier)>1: raise Exception('len(frontier)>1') # This should not happen, because find_simplex should only return one node.
        
        # Find all distorted_tri
        distorted_tri = set()
        seen = set()
        while frontier:
            f = frontier.pop()
            if f in seen:
                continue
            seen.add(f)
            a, b, c = tri.simplices[f]
            ax, ay = pts[a]; bx, by = pts[b]; cx, cy = pts[c]
            if orient2d(ax, ay, bx, by, cx, cy) < 0:    # make sure Counter-Clockwise
                b, c = c, b
                bx, by, cx, cy = cx, cy, bx, by
            if in_circumcircle(tx, ty, ax, ay, bx, by, cx, cy):
                distorted_tri.add(f)
                for nb in tri.neighbors[f]: # check f's neighbor, if t_xy in it's circumcircle.
                    if nb != -1 and nb not in seen:
                        frontier.append(nb)
                        
        if not distorted_tri:
            raise Exception("Find target node outside of covexity. Please have a check.") # Should never happen

        # For each distorted_tri, add three nodes of the tri.
        neigh_ids = set()
        for f in distorted_tri:
            a, b, c = tri.simplices[f]
            neigh_ids.add(idx2id[a])
            neigh_ids.add(idx2id[b])
            neigh_ids.add(idx2id[c])

        return neigh_ids

    def voronoi_delaunay_sources(nodes,worst_instance_greedy=False):
        '''
        ### Voronoi - Delaunay helps reduce the number of edges that need to be considered when building the MST 
        Voronoi Diagram, V(P): Divides the plane into n regions, where each side of each region is the perpendicular bisector from a point to the nearest point around it. 
        Delaunay Triangulation, D(P): Connects the points in P with edges to form triangles such that the circumcircle of any triangle contains no other points from the set P in its interior. 
        '''
        voronoi_delaunay_edges = []
        same_position_depot_list = []
        pts  = np.array([depots_nodes_position[k] for k in nodes])
        adjS = defaultdict(set)

        # Add edge for the nodes in the same position, their edge weight is 0.
        pos2nodes = {}                       
        for k in nodes:
            pos = tuple(depots_nodes_position[k])
            pos2nodes.setdefault(pos, []).append(k)

        for dup_nodes in pos2nodes.values():
            if len(dup_nodes) > 1:
                # We are using delaunay trangle to reduce the number of sources that targets need to take care. (finding the surrounding sources of each target.)
                # During the "Delaunay(pts)", if there are same location nodes, delaunay trangle only consider one of them and ignore others.
                # However, the depots located in the same location should have the same chance of being updated,
                # so we record the same position depots, and later on we add them to the sources that targets need to take care(update).
                same_position_depot_list.append(dup_nodes) 
                warnings.warn("### Warning ###: Duplicate depots found.")

        if not worst_instance_greedy:
            tri   = Delaunay(pts)                             # tri also use the first position of a dup_node
            edges = set()
            for simplex in tri.simplices:
                for i in range(3):
                    u_id, v_id = sorted((simplex[i], simplex[(i+1) % 3]))
                    if (u_id, v_id) not in edges:
                        edges.add((u_id, v_id))
                        u, v = nodes[u_id], nodes[v_id]
                        adjS[u].add(v)
                        adjS[v].add(u)
                        w = euclidean_distance(depots_nodes_position[u], depots_nodes_position[v])
                        voronoi_delaunay_edges.append((w, u, v))

            return tri, pts, adjS, voronoi_delaunay_edges, same_position_depot_list
        
        else:
            # get the left and right hand side node of each node
            order = np.argsort(pts[:, 0])  
            for i in range(len(order)):
                u_id = order[i]
                u = nodes[u_id]

                if i > 0:
                    v_id = order[i-1]
                    v = nodes[v_id]
                    adjS[u].add(v)
                    adjS[v].add(u)
                    w = euclidean_distance(depots_nodes_position[u], depots_nodes_position[v])
                    voronoi_delaunay_edges.append((w, u, v))

                if i < len(order) - 1:
                    v_id = order[i+1]
                    v = nodes[v_id]
                    adjS[u].add(v)
                    adjS[v].add(u)
                    w = euclidean_distance(depots_nodes_position[u], depots_nodes_position[v])
                    voronoi_delaunay_edges.append((w, u, v))

            return pts, adjS, voronoi_delaunay_edges, same_position_depot_list
        
    # Use target_sources_edges. 
    # When we connect new source node to the tree, we consider source--connect-to-->source and target--connect-to-->source
    
    # The distance of a source to the growing tree is determined by its minimum distance to the merged depot, referred to as super-depot, 
    # and min {MST_K * any source already in the tree, any target already in the tree}. 

    def prim_mst_ss_st(depot_to_source_edges,target_sources_edges, adjS, MST_K):
        target_of = dict(st_pairs)              # source -> target
        sources = [s for s, _ in st_pairs]
        remaining = set(sources)
        depot = 'r_super'

        # frontier initially only has depot；later on will add the connected targets
        frontier = {depot}

        best_dist = {}
        best_from = {}

        for d, p, s in depot_to_source_edges:
            if s in remaining:
                # distance between source and depot
                best_dist[s] = d
                best_from[s] = p  #'r_super'

        mst_edges = []

        heap_best_dist = [(v, 1, k) for k,v in best_dist.items()]
        heapq.heapify(heap_best_dist)
        while remaining:
            # select the source which has the min distance between 'frontiers' and 'remaining sources'
            while True:
                dist, priority, s = heapq.heappop(heap_best_dist) # priority = 0 --> target–source(high priority); # priority = 1 --> source-source(low priority)
                if s in remaining:
                    if (dist != best_dist[s]): # In heap_best_dist, if there is a duplicate (d, s2), it is a old one. We should never reach them, because there is a shorter distance for it and this shorter one should come first.
                        raise Exception('s in remaining and dist != best_dist[s], please have a check.')
                    break
            # s = min(remaining, key=lambda x: best_dist[x])
            p = best_from[s]
            w_ps = best_dist[s]
            mst_edges.append((w_ps, p, s))

            # connect to its target
            t = target_of[s]
            w_st = euclidean_distance(depots_nodes_position[s], depots_nodes_position[t])
            mst_edges.append((w_st, s, t))

            frontier.add(t)
            remaining.remove(s)

            del best_dist[s]
            del best_from[s]

            # use the new target t to update the min distance (between this target t and other remaining source)
            # update distance of sources close to t (according to Delaunay triangles that are broken) with distance to t.
            pt = depots_nodes_position[t]
            for sNearby_t in target_sources_edges[t]:
                if sNearby_t in remaining:
                    d = euclidean_distance(pt, depots_nodes_position[sNearby_t])
                    if d < best_dist[sNearby_t]:
                        best_dist[sNearby_t] = d
                        best_from[sNearby_t] = t
                        heapq.heappush(heap_best_dist, (d, 0, sNearby_t)) # push the new best_dist. old (d, s2) will still in the heap. They are always at the end, we won't reach them, because of the 'while remaining'.
            
            # update distance of sources close to s (Delaunay neighbours of s) with distance to s
            ps = depots_nodes_position[s]
            for sNearby_s in adjS[s]:
                if sNearby_s in remaining:
                    d = euclidean_distance(ps, depots_nodes_position[sNearby_s])
                    k = MST_K
                    d_new = d * k
                    if d_new < best_dist[sNearby_s]:
                        best_dist[sNearby_s] = d_new
                        best_from[sNearby_s] = s
                        heapq.heappush(heap_best_dist, (d_new, 1, sNearby_s)) # push the new best_dist. old (d, s2) will still in the heap. They are always at the end, we won't reach them, because of the 'while remaining'.
    
        return mst_edges
    
    # Use target_sources_edges. 
    # When we connect new source node to the tree, we consider target --connect-to--> source.
    # The distance of a source to the growing tree is determined by its minimum distance to the merged depot, referred to as super-depot, and any target already in the tree. 
    def prim_mst(depot_to_source_edges,target_sources_edges):
        target_of = dict(st_pairs)              # source -> target
        sources = [s for s, _ in st_pairs]
        remaining = set(sources)
        depot = 'r_super'

        # frontier initially only has depot；later on will add the connected targets
        frontier = {depot}

        best_dist = {}
        best_from = {}

        for d, p, s in depot_to_source_edges:
            if s in remaining:
                # distance between source and depot
                best_dist[s] = d
                best_from[s] = p  #'r_super'

        mst_edges = []

        heap_best_dist = [(v,k) for k,v in best_dist.items()]
        heapq.heapify(heap_best_dist)
        while remaining:
            # select the source which has the min distance between 'frontiers' and 'remaining sources'
            while True:
                dist, s = heapq.heappop(heap_best_dist)
                if s in remaining:
                    if (dist != best_dist[s]): # In heap_best_dist, if there is a duplicate (d, s2), it is a old one. We should never reach them, because there is a shorter distance for it and this shorter one should come first.
                        raise Exception('s in remaining and dist != best_dist[s], please have a check.')
                    break
            # s = min(remaining, key=lambda x: best_dist[x])
            p = best_from[s]
            w_ps = best_dist[s]
            mst_edges.append((w_ps, p, s))

            # connect to its target
            t = target_of[s]
            w_st = euclidean_distance(depots_nodes_position[s], depots_nodes_position[t])
            mst_edges.append((w_st, s, t))

            frontier.add(t)
            remaining.remove(s)

            del best_dist[s]
            del best_from[s]

            # use the new target t to update the min distance (between this target t and other remaining source)
            pt = depots_nodes_position[t]
            for s2 in target_sources_edges[t]:
                if s2 in remaining:
                    d = euclidean_distance(pt, depots_nodes_position[s2])
                    if d < best_dist[s2]:
                        best_dist[s2] = d
                        best_from[s2] = t
                        heapq.heappush(heap_best_dist, (d, s2)) # push the new best_dist. old (d, s2) will still in the heap. They are always at the end, we won't reach them, because of the 'while remaining'.

        return mst_edges

    def euclidean_distance(pos1, pos2):
        return math.hypot(pos1[0] - pos2[0], pos1[1] - pos2[1])

    def build_all_edges(points):
        edges = []
        keys = list(points.keys())
        for i in range(len(keys)):
            for j in range(i+1, len(keys)):
                u, v = keys[i], keys[j]
                dist = euclidean_distance(points[u], points[v])
                edges.append((dist, u, v))
        return edges
    
    def build_adj(edges,mst):
        adj = defaultdict(list)
        for w, u, v in edges:
            if mst: # add both direction
                adj[u].append(v)
                adj[v].append(u)
            else:   # add one direction
                adj[u].append(v)
        return adj
    
    def dfs_adj(adj, depot):
        visited, order = set(), []

        sources = {s for (s, t) in st_pairs}

        stack = [depot]

        while stack:
            u = stack.pop()
            if u in visited:
                continue
            visited.add(u)

            if u not in depots_position:
                if u in sources:
                    order.append(u)
            for v in reversed(adj.get(u, [])):
                if v not in visited:
                    stack.append(v)

        return order
    
    def route_length_add_s_t_pair(depot_pos, sources_pos: dict, source_to_target_pos: dict, visit_order):
        """according depot→source→target→… accumulated distance"""
        total = 0.0
        cur = depot_pos

        for i, s in enumerate(visit_order):
            target = source_to_target_pos[s]

            # depot/prev → source # only for depot, otherwise 0
            total += euclidean_distance(cur, sources_pos[s])
            cur = sources_pos[s]

            # source → target
            total += euclidean_distance(cur, target)
            cur = target

            # target → next source（if exist)
            if i + 1 < len(visit_order):
                nxt = visit_order[i + 1]
                total += euclidean_distance(cur, sources_pos[nxt])
                cur = sources_pos[nxt]

        return total
    import math

    def route_length(depot_pos, sources_pos: dict, source_to_target_pos: dict, visit_order):
        """according depot → source → target → … accumulated distance"""
        total = 0.0
        cur = depot_pos

        for i, s in enumerate(visit_order):
            target = source_to_target_pos[s]

            # depot/prev → source # only for depot, otherwise 0
            total += euclidean_distance(cur, sources_pos[s])
            cur = sources_pos[s]

            # source → target
            total += euclidean_distance(cur, target)
            cur = target

            # target → next source（if exist)
            if i + 1 < len(visit_order):
                nxt = visit_order[i + 1]
                total += euclidean_distance(cur, sources_pos[nxt])
                cur = sources_pos[nxt]

        return total
    
    # calculate distance of st_pairs
    def calculate_st_pair_distances(st_pairs):
        st_pair_distances = {}
        for u, v in st_pairs:
            x1, y1 = nodes_position[u]
            x2, y2 = nodes_position[v]
            d = math.hypot(x1 - x2, y1 - y2)
            st_pair_distances[u] = d
        return st_pair_distances
                
    def add_s_t_distance_to_edges(edges):
        edges_add_s_t = []
        source_node = {source for source, target in st_pairs}
        st_pair_distances = calculate_st_pair_distances(st_pairs)
        for weight, m, n in edges:
            if m in source_node: 
                weight += st_pair_distances.get(m)
            if n in source_node: 
                weight += st_pair_distances.get(n)
            edges_add_s_t.append((weight, m, n))
        return edges_add_s_t

    def combine_defaultdicts(*dicts):
        result = defaultdict(list)
        for dict in dicts:
            for k,v in dict.items():
                result[k].extend(v)
        return result
    
    def dist_cached(a, b, pos_dict, cache):
        if a == b:
            return 0.0
        key = (a, b)
        d = cache.get(key)
        if d is None:
            ax, ay = pos_dict[a]
            bx, by = pos_dict[b]
            d = math.hypot(bx - ax, by - ay)
            cache[key] = d
        return d
    
    def get_depot_source_dic_greedy(depot_source_dic,stTour_distance_dic={}):
        if stTour_distance_dic:
            st_dict = dict(stTour_distance_dic.keys())
        else:

            st_dict = dict(st_pairs)
        depot_task_dic = {key: [] for key in depots_position} # store depot and its s_t pair tasks in visiting order

        for depot, sources in depot_source_dic.items():
            depot_speed = depot_speed_dic[depot]        # for each depot
            tasks = depot_task_dic[depot]
            for sources_new in sources:
                target_new = st_dict[sources_new]
                s_new_x, s_new_y = depots_nodes_position[sources_new]
                t_new_x, t_new_y = depots_nodes_position[target_new]
                # distance of new task itself
                if stTour_distance_dic:
                    new_task_dist = stTour_distance_dic.get((sources_new,target_new))
                else:
                    new_task_dist = math.hypot(t_new_x - s_new_x, t_new_y - s_new_y)    

                depot_x, depot_y = depots_nodes_position[depot]
                depot_sNew_dist = math.hypot(s_new_x - depot_x, s_new_y - depot_y)      # distance between depot and new task
                
                # try to find the location to insert new task with the min cost
                min_insert_cost = float('inf')
                min_insert_location = {}

                if tasks == []:
                    # When there is no task in the depot, connect the new task to the depot.
                    insert_cost = (depot_sNew_dist + new_task_dist)/depot_speed
                    if insert_cost < min_insert_cost:
                        min_insert_cost = insert_cost
                        min_insert_location.clear() # To avoid storing the insertion positions of multiple depots.
                        min_insert_location[depot] = depot 
                else:
                    # When there is one or more tasks in the depot, insert the new task, between depot and first task, between tasks, or the last.                           
                    for task_i in range(len(tasks)):
                        s_x, s_y = depots_nodes_position[tasks[task_i][0]]     # task_i
                        t_x, t_y = depots_nodes_position[tasks[task_i][1]]
                        task_to_newTask_dist = math.hypot(s_new_x - t_x, s_new_y - t_y)     # distance between task_i and new task .
                        
                        # Insert the new task between the depot and the first task
                        if task_i == 0:
                            newTask_to_Task_dist = math.hypot(s_x - t_new_x, s_y - t_new_y)
                            task_to_depot_dist = math.hypot(depot_x - s_x, depot_y - s_y)
                            insert_cost = (depot_sNew_dist + new_task_dist + newTask_to_Task_dist - task_to_depot_dist)/depot_speed
                            if insert_cost < min_insert_cost:
                                min_insert_cost = insert_cost
                                min_insert_location.clear()
                                min_insert_location[depot] = depot 

                        # Insert the new task between tasks or the last
                        if task_i == len(tasks)-1:      # last task
                            insert_cost = (task_to_newTask_dist + new_task_dist)/depot_speed
                            if insert_cost < min_insert_cost:
                                min_insert_cost = insert_cost
                                min_insert_location.clear()
                                min_insert_location[depot] = task_i 
                        else:
                            n_s_x, n_s_y = depots_nodes_position[tasks[task_i+1][0]]           # task i+1
                            n_t_x, n_t_y = depots_nodes_position[tasks[task_i+1][1]]
                            newTask_to_nextTask_dist = math.hypot(t_new_x - n_s_x, t_new_y - n_s_y)
                            task_to_nextTask_dist = math.hypot(n_s_x - t_x, n_s_y - t_y)
                            insert_cost = (task_to_newTask_dist + new_task_dist + newTask_to_nextTask_dist - task_to_nextTask_dist)/depot_speed
                            if insert_cost < min_insert_cost:
                                min_insert_cost = insert_cost
                                min_insert_location.clear()
                                min_insert_location[depot] = task_i 
                # insert new st_pair
                for depot, insert_position in min_insert_location.items():
                    tasks = depot_task_dic[depot]
                    if insert_position == depot: 
                        tasks.insert(0,(sources_new,target_new))
                    else:
                        for t in range(len(tasks)):
                            if t == insert_position:
                                tasks.insert((t+1), (sources_new,target_new))
                                break
        return depot_task_dic

    def depot_task_dic_to_final_cost(depot_task_dic, stTour_distance_dic={}):
        total_cost = 0
        total_distance = 0
        use_depot_count = 0
        depot_st_tour_dist_dic = {}
        for depot, tasks in depot_task_dic.items():
            depot_x, depot_y = depots_nodes_position[depot]
            depot_speed = depot_speed_dic[depot]
            depot_task_dist = float('inf')
            distance_each_depot = 0
            firstSource_to_lastTarget_distance = 0
            for task_i in range(len(tasks)):
                s_x, s_y = depots_nodes_position[tasks[task_i][0]]     # task i
                t_x, t_y = depots_nodes_position[tasks[task_i][1]]
                if stTour_distance_dic:
                    task_dist = stTour_distance_dic.get((tasks[task_i][0],tasks[task_i][1]))
                else:
                    task_dist = math.hypot(s_x - t_x, s_y - t_y)
                # distance of task_i
                distance_each_depot += task_dist
                # distance from depot to task_1
                if task_i == 0:
                    depot_task_dist = math.hypot(s_x - depot_x, s_y - depot_y)
                    distance_each_depot += depot_task_dist
                # distance between tasks (until connect with the last task. Note: len(tasks)-1 is the index of last task_i)
                if task_i < len(tasks)-1: 
                    n_s_x, n_s_y = depots_nodes_position[tasks[task_i+1][0]]     # task i
                    between_tasks_dist = math.hypot(n_s_x - t_x, n_s_y - t_y)
                    distance_each_depot += between_tasks_dist
            total_cost += distance_each_depot/depot_speed
            total_distance += distance_each_depot
            
            # record the tour distance of a tree
            if len(tasks)>0:
                if depot_task_dist ==float('inf'): raise Exception('len(tasks)>0, but depot_task_dist is inf. Please have a check.')
                firstSource_to_lastTarget_distance = distance_each_depot - depot_task_dist
                # {depot: ((first task's source, last task's target), distance between them)} # each depot has a greedy tour
                depot_st_tour_dist_dic[depot] = ((tasks[0][0], tasks[len(tasks)-1][1]), firstSource_to_lastTarget_distance)

            # record the number of depot used
            if distance_each_depot > 0: use_depot_count += 1

            # print(f'Final_cost_by_final_depot_and_stPair_greedy, depot: {depot}, distance: {distance_each_depot}')
        return use_depot_count, total_distance, total_cost, depot_st_tour_dist_dic
    
    def calculate_final_cost_by_final_trees_eachTreeGreedy(pruned_results, trees_structure, greedyTrees):
        depot_source_dic = trees_structure_to_depot_source_dic(trees_structure)               
        trees_structure_greedy = get_depot_source_dic_greedy(depot_source_dic) # greedy for each small tree
        if greedyTrees:
            _, _, _, depot_st_tour_dist_dic = depot_task_dic_to_final_cost(trees_structure_greedy) # calculate small tree tour distance
            depot_source_tour_dic,stTour_distance_dic = get_depot_source_dic_forGreedyTrees(depot_st_tour_dist_dic) # prepare the depot & source for the greedyTrees 
            depot_task_dic = get_depot_source_dic_greedy(depot_source_tour_dic,stTour_distance_dic)   # greedyTrees 
            use_depot_count, total_distance, total_cost,_ = depot_task_dic_to_final_cost(depot_task_dic,stTour_distance_dic) # calculate greedyTrees tour distance
            print(f'final_tour_eachTreeGreedy_greedyTrees, Number of depot used: {use_depot_count}')
            # print(f'final_tour_eachTreeGreedy_greedyTrees, Final distance: {total_distance}')
            print(f'final_tour_eachTreeGreedy_greedyTrees, Final cost: {total_cost}')
            return total_cost
        else: 
            flattened_trees_structure_greedy = flattened_dict(trees_structure_greedy)
            return calculate_final_cost_by_final_trees_eachTree(pruned_results,flattened_trees_structure_greedy)

        
    # connect trees, use each tree's node order 
    def calculate_final_cost_by_final_trees_eachTree(pruned_results,trees_structure):
        final_cost = 0
        final_distance = 0
        use_depot_count = 0

        # get the edges from tree combination. split mst_result_trees according to depot. 
        # step1: get depot_sources dic, step2: get edges of each source, then we get depot_edges dic
        depot_source_dic = get_depot_source_dic(pruned_results,trees_structure)
        #e.g., defaultdict(<class 'list'>, {'1.2': ['5'], '1.1': ['3', '1', '7', '13', '15', '9', '11']})

        # print(f'time 3: {time.time()}')

        for depot, sources in depot_source_dic.items():
            depot_speed = depot_speed_dic[depot]
            src_order = sources     # visit the trees connected with this depot. Iterate one tree at a time, according to the DFS of the MST (from tree combination) 
            sources_pos = {source: depots_nodes_position[source] for source, _ in st_pairs}
            source_to_target_pos = {source: depots_nodes_position[target] for source, target in st_pairs}
            
            # sources_pos and source_to_target_pos include more source than requred, it doesn't matter, because route_length will be calculated according to src_order, only the sources in src_order will be counted.
            total_distance = route_length(depots_nodes_position[depot], sources_pos, source_to_target_pos, src_order)
            
            total_cost = total_distance/depot_speed
            final_distance += total_distance
            final_cost += total_cost
            # record the number of depot used
            if total_distance > 0: use_depot_count += 1

            # print(f"depot:{depot}, depot_speed: {depot_speed}, total distance: {total_distance:.2f}, total cost: {total_cost:.2f}")

        # print(f'time 4: {time.time()}')

        print(f'Number of depot used: {use_depot_count}')
        # print(f'Final distance: {final_distance}')
        print(f"Final cost (time): {final_cost}")
        return final_cost

    def calculate_final_cost_by_final_trees(pruned_results,new_edges_set,trees_structure,tree_distances,mst_result_trees):
        final_cost = 0
        final_distance = 0
        use_depot_count = 0
        # get edges between trees
        edge_list_dic = get_depot_edges_dic(pruned_results,new_edges_set)
        # get real connnected nodes and distances between trees
        edge_list_dic_depots = defaultdict(list)
        for depot, edges in edge_list_dic.items():
            edges = [
                ((tree_distances[(u, v)]['distance'], tree_distances[(u, v)]['node_from_A'],tree_distances[(u, v)]['node_from_B']) if (u, v) in tree_distances else (tree_distances[(v, u)]['distance'], tree_distances[(v, u)]['node_from_A'], tree_distances[(v, u)]['node_from_B']))
                for (u, v) in edges] 
            edge_list_dic_depots[depot] = edges

        # get the edges from tree combination. split mst_result_trees according to depot. 
        # step1: get depot_sources dic, step2: get edges of each source, then we get depot_edges dic
        depot_source_dic = get_depot_source_dic(pruned_results,trees_structure)
        mst_result_trees_depots = defaultdict(list)

        # A depot tree may comtain multple trees, we need to count all their edges.
        # Since we know all the sources connected with this depot tree, we can just find the source's corresponding edges.
        
        # Find the source's corresponding edges: {node: list of (weight, u, v)}
        node_to_edges = defaultdict(list)
        for u, v, w in mst_result_trees:
            node_to_edges[u].append((w, u, v))
            if u != v:
                node_to_edges[v].append((w, u, v))

        mst_result_trees_depots = defaultdict(list)

        # For each depot tree, search the edges' of each source.
        for depot, sources in depot_source_dic.items():
            seen_edges = set()
            for node in sources:
                for edge in node_to_edges.get(node, []):
                    edge_id = tuple(sorted([edge[1], edge[2]]))  # sort the u,v, for (u,v),(v,u) duplicate
                    if edge_id not in seen_edges:
                        seen_edges.add(edge_id)
                        mst_result_trees_depots[depot].append(edge)
 
        combined_edges_dic = combine_defaultdicts(edge_list_dic_depots, mst_result_trees_depots)
        
        for depot, edges in combined_edges_dic.items():
            depot_speed = depot_speed_dic[depot]
            adj = build_adj(edges, mst=True)
            src_order = dfs_adj(adj, depot)
            if len(src_order) != len(depot_source_dic.get(depot)): 
                raise Exception('Did not go though all nodes of this depot.')
            sources_pos = {source: depots_nodes_position[source] for source, _ in st_pairs}
            source_to_target_pos = {source: depots_nodes_position[target] for source, target in st_pairs}
            # sources_pos and source_to_target_pos include more source than requred, it doesn't matter, because route_length will be calculated according to src_order, only the sources in src_order will be counted.
            total_distance = route_length(depots_nodes_position[depot], sources_pos, source_to_target_pos, src_order)
            total_cost = total_distance/depot_speed
            final_distance += total_distance
            final_cost += total_cost
            # record the number of depot used
            if total_distance > 0: use_depot_count += 1

            # print(f"depot:{depot}, depot_speed: {depot_speed}, total distance: {total_distance:.2f}, total cost: {total_cost:.2f}")

        print(f'Number of depot used: {use_depot_count:.2f}')
        print(f'Final distance: {final_distance:.2f}')
        print(f"Final cost (time): {final_cost:.2f}")
        return final_cost
    
    def calculate_final_cost_by_final_trees_greedy(pruned_results,trees_structure):
        depot_source_dic = get_depot_source_dic(pruned_results,trees_structure)
        depot_task_dic = get_depot_source_dic_greedy(depot_source_dic)    
        use_depot_count, _, total_cost,_ = depot_task_dic_to_final_cost(depot_task_dic)

        print(f'Final_cost_by_final_depot_and_stPair_greedy, Number of depot used: {use_depot_count}')
        # print(f'Final_cost_by_final_depot_and_stPair_greedy, Final distance: {total_distance}')
        print(f'Final_cost_by_final_depot_and_stPair_greedy, Final cost: {total_cost}')
        return total_cost
    
    # e.g., [('3', '4'), ('1', '2')] to ['3', '4', '1', '2']
    def flattened_dict(dict):
        return {
            k: [x for edge in v for x in edge] for k, v in dict.items()
        }

    def save_all_seeds_to_csv(filename, all_results):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(script_dir, "plot_output", filename)
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            if not is_Greedy:
                writer.writerow([
                    "seed", "x_axle", 
                    "Tree Construction Time",        
                    "PD Initialization Time",       
                    "PD Solve Time",                 
                    "Route time",
                    "Total Time",
                    "Route Cost"
                ])
            else:
                writer.writerow([
                    "seed", "x_axle",
                    "Greedy Time",                                   
                    "final_calculation_time",
                    "Total Running time",
                    "Route Cost"
                ])


            for seed, data in all_results.items():
                for i in range(len(data["x_axle"])):
                    if not is_Greedy:
                        writer.writerow([
                            seed,
                            data["x_axle"][i],
                            data["CT"][i],        
                            data["PD_init"][i],   
                            data["PD_solve"][i],  
                            data["total"][i] - data["CT"][i] - data["PD_init"][i] - data["PD_solve"][i], # Route time
                            data["total"][i],
                            data["route_cost"][i],
                        ])
                    else:
                        writer.writerow([
                            seed,
                            data["x_axle"][i],
                            data["total"][i],
                            data["final_times"][i],
                            data["final_times"][i] + data["total"][i],
                            data["route_cost"][i],
                        ])

        print(f"Saved to CSV: {csv_path}")

    def stable_hash(depot_id: str):
        level, index = depot_id.split(".")
        return int(level) * 10000 + int(index)
        # e.g.,
        # '1.1' seed = 10000 + 1 = 10001
        # '1.10' seed = 10000 + 10 = 10010
        # '2.10' seed = 20000 + 10 = 20010

    def parse_arguments():
        parser = argparse.ArgumentParser(description="Routing experiment runner")

        # ------------------ Mandatory Inputs ------------------
        parser.add_argument(
            "--dataset_input",
            type=str,
            required=True,
            choices=[
                "uniform_distribution_data",
                "GMM_data",
                "meituan_dataset_weekday",
                "meituan_dataset_weekdend_day",
                "worst_instance_greedy",
            ],
            help="Dataset selector",
        )

        parser.add_argument(
            "--routing_method",
            type=str,
            required=True,
            choices=["PD_DFS", "PD_Greedy", "PD_DGreedy", "Greedy_baseline"],
            help="Routing method",
        )

        # constant_MST
        parser.add_argument(
            "--use_constant_mst",
            type=int,
            choices=[0, 1],
            help="0 = disable MST constraint, 1 = enable",
        )

        parser.add_argument(
            "--mst_multiplier",
            type=int,
            help="If MST enabled, multiplier k (e.g., 7). If disabled, must be 0.",
        )

        # ------------------ dataset parameters ------------------

        parser.add_argument("--st_pair_size_input", type=int, default=None)
        parser.add_argument("--depot_size_input", type=int, default=None)
        parser.add_argument("--level_size_input", type=int, default=None)

        parser.add_argument(
            "--seed_value_list_input",
            type=str,
            default=None,
            help='Comma-separated seeds, e.g. "1,2,3,4.5"',
        )

        parser.add_argument("--GMM_cluster_input", type=int, default=None)
        parser.add_argument("--GMM_sigma_input", type=float, default=None)

        args = parser.parse_args()

        # ----------- seeds -----------
        if args.seed_value_list_input is not None:
            seed_list = [
                int(x.strip()) for x in args.seed_value_list_input.split(",") if x.strip()
            ]
        else:
            seed_list = None

        # ----------- constant_MST -----------
        constant_MST = None
        if args.use_constant_mst is not None and args.mst_multiplier is not None:
            if args.use_constant_mst == 0:
                constant_MST = [False, 0]
            else:
                constant_MST = [True, args.mst_multiplier]

        return (
            args.dataset_input,
            args.routing_method,
            constant_MST,
            args.st_pair_size_input,
            args.depot_size_input,
            args.level_size_input,
            seed_list,
            args.GMM_cluster_input,
            args.GMM_sigma_input,
        )
    
    def _validate_seed_list(dataset_input, seed_value_list_input):
        if dataset_input in ["meituan_dataset_weekday",
                            "meituan_dataset_weekdend_day",
                            "worst_instance_greedy"]:
            if seed_value_list_input is None:
                return [0]
        
        # uniform and GMM must provide
        if seed_value_list_input is None:
            raise ValueError(
                f"{dataset_input} requires seed_value_list_input."
            )

        if not isinstance(seed_value_list_input, (list, tuple)):
            raise ValueError("seed_value_list_input must be a list of integers.")

        if len(seed_value_list_input) == 0:
            raise ValueError("seed_value_list_input cannot be empty.")

        clean_seeds = []
        for s in seed_value_list_input:
            if not isinstance(s, int):
                raise ValueError(f"Seed {s} is not an integer.")
            if s < 0:
                raise ValueError(f"Seed {s} must be >= 0.")
            clean_seeds.append(s)

        # Remove duplicate seeds and sort the remaining seeds.
        clean_seeds = sorted(set(clean_seeds))

        return clean_seeds

    def validate_inputs(
        dataset_input,
        routing_method,
        constant_MST,
        st_pair_size_input=None,
        depot_size_input=None,
        level_size_input=None,
        seed_value_list_input=None,
        GMM_cluster_input=None,
        GMM_sigma_input=None,
    ):
        """
        Validate required parameters based on:
        dataset_input:
            - uniform_distribution_data
            - GMM_data
            - meituan_dataset_weekday
            - meituan_dataset_weekdend_day
            - worst_instance_greedy
        routing_method:
            - PD_DFS, PD_Greedy, PD_DGreedy, Greedy_baseline
        constant_MST:
            - [False,0] or [True,k] (k>=1)

        Raises:
        ValueError with a clear message if something is missing/invalid.
        """

        # -------- validate dataset_input --------
        valid_datasets = {
            "uniform_distribution_data",
            "GMM_data",
            "meituan_dataset_weekday",
            "meituan_dataset_weekdend_day",
            "worst_instance_greedy",
        }
        if dataset_input not in valid_datasets:
            raise ValueError(
                f"Invalid dataset_input={dataset_input!r}. Must be one of {sorted(valid_datasets)}."
            )

        # -------- validate routing_method --------
        valid_methods = {"PD_DFS", "PD_Greedy", "PD_DGreedy", "Greedy_baseline"}
        if routing_method not in valid_methods:
            raise ValueError(
                f"Invalid routing_method={routing_method!r}. Must be one of {sorted(valid_methods)}."
            )

        # -------- validate constant_MST --------
        if routing_method != "Greedy_baseline":
            if not isinstance(constant_MST, (list, tuple)) or len(constant_MST) != 2:
                raise ValueError("use_constant_mst and mst_multiplier are required..")

            use_flag, mst_mult = constant_MST[0], constant_MST[1]

            if not isinstance(mst_mult, int):
                raise ValueError(f"mst_multiplier must be int, got {type(mst_mult).__name__}.")

            if use_flag is False:
                if mst_mult != 0:
                    raise ValueError("If MST is disabled (use_constant_mst=0), mst_multiplier must be 0.")
            else:
                if mst_mult < 1:
                    raise ValueError("If MST is enabled (use_constant_mst=1), mst_multiplier must be >= 1 (e.g., 7).")

        # -------- helpers for required params --------
        def _req_positive(name, val):
            if val is None:
                raise ValueError(f"Missing required parameter for dataset {dataset_input}: {name} is None.")
            if val < 0:
                raise ValueError(f"{name} must be > 0, got {val}.")
            return val

        def _req_positive_int(name, val):
            _req_positive(name, val)
            if not isinstance(val, int):
                raise ValueError(f"{name} must be an int, got {type(val).__name__}.")
            if val <= 0:
                raise ValueError(f"{name} must be > 0, got {val}.")
            return val

        def _req_level(name, val):
            _req_positive(name, val)
            if not isinstance(val, int):
                raise ValueError(f"{name} must be an int, got {type(val).__name__}.")
            if val < 1:
                raise ValueError(f"{name} must be >= 1, got {val}.")
            return val

        # -------- seeds normalization --------
        seed_list = _validate_seed_list(dataset_input, seed_value_list_input)

        # -------- dataset-specific required parameters --------
        if dataset_input == "uniform_distribution_data":
            _req_positive_int("st_pair_size_input", st_pair_size_input)
            _req_positive_int("depot_size_input", depot_size_input)
            _req_level("level_size_input", level_size_input)

        elif dataset_input == "GMM_data":
            _req_positive_int("GMM_cluster_input", GMM_cluster_input)
            _req_positive("GMM_sigma_input", GMM_sigma_input)
            if not isinstance(GMM_sigma_input, (int, float)):
                raise ValueError(f"GMM_sigma_input must be a number, got {type(GMM_sigma_input).__name__}.")
            if float(GMM_sigma_input) <= 0:
                raise ValueError(f"GMM_sigma_input must be > 0, got {GMM_sigma_input}.")
            _req_positive_int("st_pair_size_input", st_pair_size_input)
            _req_positive_int("depot_size_input", depot_size_input)
            _req_level("level_size_input", level_size_input)

        elif dataset_input in {"meituan_dataset_weekday", "meituan_dataset_weekdend_day"}:
            _req_positive_int("depot_size_input", depot_size_input)
            _req_level("level_size_input", level_size_input)
            # st_pair_size_input not required; seed is [0].

        elif dataset_input == "worst_instance_greedy":
            _req_positive_int("st_pair_size_input", st_pair_size_input)
            # depot/level not required; seed is [0].

        return {
            "seed_value_list_input": seed_list,
        }

    (
        dataset_input,
        routing_method_input,
        constant_MST_input,
        st_pair_size_input,
        depot_size_input,
        level_size_input,
        seed_value_list_input,
        GMM_cluster_input,
        GMM_sigma_input,
    ) = parse_arguments()

    validated = validate_inputs(
        dataset_input=dataset_input,
        routing_method=routing_method_input,
        constant_MST=constant_MST_input,
        st_pair_size_input=st_pair_size_input,
        depot_size_input=depot_size_input,
        level_size_input=level_size_input,
        seed_value_list_input=seed_value_list_input,
        GMM_cluster_input=GMM_cluster_input,
        GMM_sigma_input=GMM_sigma_input,
    )
    seed_value_list_input = validated["seed_value_list_input"]

    ############################################################################## 
    # 1. select dataset_input: "uniform_distribution_data", "GMM_data", "meituan_dataset_weekday","meituan_dataset_weekdend_day", "worst_instance_greedy"
    #    parameters:   
    #       "uniform_distribution_data": st_pair_size_input, depot_size_input, level_size_input, seed_value_list_input
    #       "GMM_data": GMM_cluster_input, GMM_sigma_input, st_pair_size_input, depot_size_input, level_size_input, seed_value_list_input
    #       "meituan_dataset": depot_size_input, level_size_input
    #       "worst_instance_greedy": No parameter is required.
    #    
    # 2. select routing_method: 'PD_DFS', 'PD_Greedy', 'PD_DGreedy', "Greedy_baseline"
    #    if the dataset is one of the "uniform_distribution_data", "GMM_data", "meituan_dataset"
    # 
    # 3. select constant_MST or not: [True, any number(e.g.,7)] or [False,0] 
    # 4. input seed
    
    ################# Setup ST-Pairs/ Depots/ Levels/ Seed number #################    
    is_uniform = False
    is_GMM = False
    worst_instance_greedy = False
    is_realworld_dataset = False

    if dataset_input == "uniform_distribution_data":
        is_uniform = True
        num_nodes = 2 * st_pair_size_input
        total_depots = depot_size_input
        level_k = level_size_input
        seed_value_list = seed_value_list_input

    elif dataset_input == "GMM_data":
        is_GMM = True
        GMM_cluster = GMM_cluster_input
        GMM_sigma = GMM_sigma_input
        num_nodes = 2 * st_pair_size_input
        total_depots = depot_size_input
        level_k = level_size_input
        seed_value_list = seed_value_list_input

    elif dataset_input == "worst_instance_greedy":
        worst_instance_greedy = True
        num_nodes = 2 * st_pair_size_input
        total_depots = 2
        level_k = 2
        seed_value_list = [0]

    elif dataset_input in ["meituan_dataset_weekday","meituan_dataset_weekdend_day"]:
        is_realworld_dataset = True
        if dataset_input == "meituan_dataset_weekday":
            num_nodes = 69103
        else:
            num_nodes = 77638
        total_depots = depot_size_input
        level_k = level_size_input
        seed_value_list = [0]


    constant_MST = constant_MST_input # Make sure the route is in constant time of MST --> constant_MST = [True,k]

    routing_method = routing_method_input
    
    if routing_method == 'Greedy_baseline':
        is_Greedy = True
    else: 
        is_Greedy = False

    all_results = {} 

    for seed_value in seed_value_list:
        ### Record running time and route cost ###
        CT_times = []
        initial_algorithm_times  = []
        algorithm_times = []
        total_times = []
        route_cost_list = []
        final_times = []

        base = total_depots // level_k
        rem = total_depots % level_k
        K_list = [base + 1 if i < rem else base for i in range(level_k)]

        depots, levels, depot_speed_dic = generate_levels(level_k, K_list)
        levels_copy = copy.deepcopy(levels)

        start_total = time.time()
        # 1. Get depot positions 
        # 2. Get nodes positions
        # 3. Get ST pairs based on the nodes
                                    
        if is_uniform:
            # Generate depot
            depots_position = {}
            for depot in depots:
                rng_i = random.Random(seed_value + stable_hash(depot))  # 每个 depot 自己的随机数
                depots_position[depot] = (rng_i.uniform(0, 100), rng_i.uniform(0, 100))

            # Generate nodes
            rng = random.Random(seed_value)
            nodes_position = {str(i): (rng.uniform(0, 100), rng.uniform(0, 100)) for i in range(1, num_nodes + 1)}
            # Generate random ST pairs based on the nodes and the number of node
            # n nodes can generate n/2 s-t pair. (to avoid same s or same t name)
            st_pairs = generate_st_pairs(nodes_position, int(num_nodes/2))

        elif is_GMM:
            # Generate nodes and depots
            # clusters，sigma，region=[0,100]^2
            nodes_position, depots_position, centers, ks= sample_gmm_nodes(num_nodes, depots, Z=GMM_cluster, L=100, sigma=GMM_sigma, seed=seed_value)
            
            # Generate ST pairs; 70% clustered, distance between [2, 25]
            # st_pairs = generate_st_pairs_gmm(
            #     nodes_position, labels, num_pairs=int(num_nodes/2),
            #     same_cluster_p=0.7, dmin=2.0, dmax=25.0
            # )
            st_pairs = generate_st_pairs(nodes_position, int(num_nodes/2))                                
        
        elif worst_instance_greedy:
            n = int(num_nodes/2)
            epsilon = 0.01
            v1 = 1000
            v2 = 1
            v1_x = -(n+epsilon)*v1
            v2_x = -v2*n 

            depots_position = {'1.1': (v1_x,0),
                                '2.1': (v2_x,0)}
            nodes_position = {}
            st_pairs = []
            spacing = int(num_nodes/2)*v2

            # source nodes: 0 ~ n-1
            for i in range(int(num_nodes/2)):
                nodes_position[str(i)] = (i * spacing, 0)

            # target nodes: n ~ 2n-1
            for i in range(int(num_nodes/2)):
                nodes_position[str(i + int(num_nodes/2))] = (i * spacing, 0)
                st_pairs.append((str(i), str(i + int(num_nodes/2))))
        
        elif is_realworld_dataset:
            BASE_DIR = Path(__file__).resolve().parent
            
            # depots
            depots_file_path = BASE_DIR / "meituan_data" / "meituan_couriers_position" / f"{total_depots}_couriers_position.pkl"
            with open(depots_file_path, "rb") as f:
                depots_data = pickle.load(f)

            depots_position = {}
            nodes_position = {}
            st_pairs = []
            depots_position = {}

            assert len(depots) == len(depots_data)

            depots_position = dict(zip(depots, depots_data.values())) # Fill the depot location into the depots.

            # ST Pairs, "weekday_18_stPairs": 69103, "weekend_22_stPairs":77638
            if dataset_input == "meituan_dataset_weekday": 
                stPairs_file_path = BASE_DIR / "meituan_data" / "weekday_18_stPairs.pkl" 
            else:
                stPairs_file_path = BASE_DIR / "meituan_data" / "weekend_22_stPairs.pkl" 
            
            with open(stPairs_file_path, "rb") as f:
                stPairs_data = pickle.load(f)

            nodes_position = stPairs_data.get("nodes_position")

            st_pairs = stPairs_data.get("st_pairs")

        else:
            raise Exception('Please select the dataset you want.')
        
        # Combine depot position and node position
        depots_nodes_position = depots_position | nodes_position

        # plot
        # plot_graph(depots_position,nodes_position,st_pairs,103)

        
        if is_Greedy:
            start_Greedy = time.time()

            depot_task_dic = {key: [] for key in depots_position}
            for i in range(len(st_pairs)):
                s_new_x, s_new_y = depots_nodes_position[st_pairs[i][0]]
                t_new_x, t_new_y = depots_nodes_position[st_pairs[i][1]]
                new_task_dist = math.hypot(t_new_x - s_new_x, t_new_y - s_new_y)
                min_insert_cost = float('inf')
                min_insert_location = {}
                # find the place to insert new st_pair
                for depot, tasks in depot_task_dic.items():
                    depot_speed = depot_speed_dic[depot]
                    depot_x, depot_y = depots_nodes_position[depot]
                    depot_sNew_dist = math.hypot(s_new_x - depot_x, s_new_y - depot_y)
                    if tasks == []:
                        # When there is no task in the depot, connect the new task to the depot.
                        insert_cost = (depot_sNew_dist + new_task_dist)/depot_speed
                        if insert_cost < min_insert_cost:
                            min_insert_cost = insert_cost
                            min_insert_location.clear() # To avoid storing the insertion positions of multiple depots.
                            min_insert_location[depot] = depot 
                    else:
                        # When there is one or more tasks in the depot, insert the new task, between depot and first task, between tasks, or the last.                           
                        for task_i in range(len(tasks)):
                            s_x, s_y = depots_nodes_position[tasks[task_i][0]]     # task i
                            t_x, t_y = depots_nodes_position[tasks[task_i][1]]
                            task_to_newTask_dist = math.hypot(s_new_x - t_x, s_new_y - t_y)     # distance between new task and task i.
                            
                            # Insert the new task between the depot and the first task
                            if task_i == 0:
                                newTask_to_Task_dist = math.hypot(s_x - t_new_x, s_y - t_new_y)
                                task_to_depot_dist = math.hypot(depot_x - s_x, depot_y - s_y)
                                insert_cost = (depot_sNew_dist + new_task_dist + newTask_to_Task_dist - task_to_depot_dist)/depot_speed
                                if insert_cost < min_insert_cost:
                                    min_insert_cost = insert_cost
                                    min_insert_location.clear()
                                    min_insert_location[depot] = depot 

                            # Insert the new task between tasks or the last
                            if task_i == len(tasks)-1:      # last task
                                insert_cost = (task_to_newTask_dist + new_task_dist)/depot_speed
                                if insert_cost < min_insert_cost:
                                    min_insert_cost = insert_cost
                                    min_insert_location.clear()
                                    min_insert_location[depot] = task_i 
                            else:
                                n_s_x, n_s_y = depots_nodes_position[tasks[task_i+1][0]]           # task i+1
                                n_t_x, n_t_y = depots_nodes_position[tasks[task_i+1][1]]
                                newTask_to_nextTask_dist = math.hypot(t_new_x - n_s_x, t_new_y - n_s_y)
                                task_to_nextTask_dist = math.hypot(n_s_x - t_x, n_s_y - t_y)
                                insert_cost = (task_to_newTask_dist + new_task_dist + newTask_to_nextTask_dist - task_to_nextTask_dist)/depot_speed
                                if insert_cost < min_insert_cost:
                                    min_insert_cost = insert_cost
                                    min_insert_location.clear()
                                    min_insert_location[depot] = task_i 
                # insert new st_pair
                for depot, insert_position in min_insert_location.items():
                    tasks = depot_task_dic[depot]
                    if insert_position == depot: 
                        tasks.insert(0,st_pairs[i])
                    else:
                        for t in range(len(tasks)):
                            if t == insert_position:
                                tasks.insert((t+1), st_pairs[i])
                                break
            
            end_Greedy = time.time()
            # depot_task_dic 
            # '1.1' = [('1', '2'), ('3', '4'), ('5', '6')]
            # '1.2' = [('7', '8')]
            # '2.1' = [('9', '10')]
            # '2.2' = []
            total_cost = 0
            total_distance = 0
            use_depot_count = 0
            for depot, tasks in depot_task_dic.items():
                depot_x, depot_y = depots_nodes_position[depot]
                depot_speed = depot_speed_dic[depot]
                distance_each_depot = 0
                for task_i in range(len(tasks)):
                    s_x, s_y = depots_nodes_position[tasks[task_i][0]]     # task i
                    t_x, t_y = depots_nodes_position[tasks[task_i][1]]
                    task_dist = math.hypot(s_x - t_x, s_y - t_y)
                    # distance of task_i
                    distance_each_depot += task_dist
                    # distance from depot to task_1
                    if task_i == 0:
                        depot_task_dist = math.hypot(s_x - depot_x, s_y - depot_y)
                        distance_each_depot += depot_task_dist
                    # distance between tasks (until connect with the last task. Note: len(tasks)-1 is the index of last task_i)
                    if task_i < len(tasks)-1: 
                        n_s_x, n_s_y = depots_nodes_position[tasks[task_i+1][0]]     # task i
                        n_t_x, n_t_y = depots_nodes_position[tasks[task_i+1][1]]
                        between_tasks_dist = math.hypot(n_s_x - t_x, n_s_y - t_y)
                        distance_each_depot += between_tasks_dist
                total_cost += distance_each_depot/depot_speed
                total_distance += distance_each_depot

                # record the number of depot used
                if distance_each_depot > 0: use_depot_count += 1

                # print(f'Greedy algorithm, depot: {depot}, distance: {distance_each_depot}')
            
            final_Greedy = time.time()

            print(f'Greedy algorithm, Number of depot used: {use_depot_count}')
            # print(f'Greedy algorithm, Final distance: {total_distance}')
            print(f'Greedy algorithm, Final cost: {total_cost}')
            total_time = end_Greedy - start_Greedy
            print(f'Greedy algorithm, Running time: {total_time}')

            route_cost_list.append(total_cost)
            total_times.append(total_time)
            
            final_time = final_Greedy - end_Greedy
            final_times.append(final_time)

        else:
            # Tree Construction
            # start_CT
            start_CT = time.time()

            ConstructTree = ConstructTreeProblem(depots_nodes_position, depots, st_pairs)
            mst_result_trees, source_target_edge = ConstructTree.construct_mst_with_super_depot(constant_MST, worst_instance_greedy)
            # constructTree_timing_log = ConstructTree.step_timing_log
            final_trees = ConstructTree.build_trees_from_prim_result(mst_result_trees)
            
            # Plot MST trees 
            # ConstructTree.plot_trees_plotly(mst_result_trees, 103)
            
            # Weight of each tree
            weights = {key: value[1] for key, value in final_trees.items()}
            # The constructed trees
            trees_structure = {key: value[0] for key, value in final_trees.items()}
            
            # end_CT
            end_CT = time.time()
            CT_times.append(end_CT - start_CT)

            # start_initial_algorithm_times
            start_initial_algorithm = time.time()
            
            # Primal_Dual Algorithm
            algorithm = PrimalDualAlgorithm(levels, depots_nodes_position, trees_structure, weights)
            
            # end_initial_algorithm_time
            end_initial_algorithm = time.time()
            
            initial_algorithm_times.append(end_initial_algorithm-start_initial_algorithm)

            # start_alg
            start_alg = time.time()
            
            algorithm.run_algorithm()
            # timing_log = algorithm.step_timing_log
            # compute_max_dual_growth_time.append(sum(step.get("compute_max_dual_growth_time") for step in timing_log))
            # grow_dual_variables_time.append(sum(step.get("grow_dual_variables_time") for step in timing_log))
            # update_components_time.append(sum(step.get("update_components_time") for step in timing_log))
            # total_step_time.append(sum(step.get("total_step_time") for step in timing_log))
            # capture_step_time.append(sum(step.get("capture_step_time") for step in timing_log))
            # const_7_time.append(sum(step.get("C7_time") for step in timing_log))
            # const_8_time.append(sum(step.get("C8_time") for step in timing_log))

            new_edges_set = algorithm.get_set_from_dic(algorithm.new_edges)
            tree_edge_costs = algorithm.edge_costs.copy()
            
            historical_frozen_component = algorithm.get_set_from_dic(algorithm.previously_frozen_nodes)
            # print("historical_frozen_component", dict(algorithm.previously_frozen_nodes))

            # start_pruning = time.time()

            # Pruning Procedure 
            pruned_results = algorithm.prune_forests(levels, algorithm.active_components.copy(), new_edges_set, historical_frozen_component)
            
            ###  Check if all depots appear in pruned results ###
            # Flatten all nodes from pruned_results
            all_nodes_in_pruned = set()
            for comps in pruned_results.values():
                for comp, _ in comps:
                    all_nodes_in_pruned.update(comp)
            nonzerodepots_position = {k:v for k,v in weights.items() if v > 0}
            missing_depots = [d for d in nonzerodepots_position if d not in all_nodes_in_pruned]
            if missing_depots:
                raise Exception(f"Missing depots in pruned results: {missing_depots}. Please have a check")
            
            # end_alg
            end_alg = time.time()

            algorithm_times.append(end_alg - start_alg)
            print(f'TC+P_d algorithm running time: {end_alg- start_total}')

                
            if routing_method == 'PD_Greedy':
                ## P_D tree: greedy ###
                print('---------------- P_D tree: greedy ----------------')
                route_cost = calculate_final_cost_by_final_trees_greedy(pruned_results,trees_structure)
            elif routing_method == 'PD_DGreedy':
                ## TC's each tree: greedy, P_D tree greedy ###
                print("---------------- TC's each tree: initial order (DFS), then greedy, P_D tree greedy ----------------")
                route_cost = calculate_final_cost_by_final_trees_eachTreeGreedy(pruned_results, trees_structure, greedyTrees = True)
            elif routing_method == "PD_DFS":
                ## P_D tree: DFS ###
                print('---------------- calculate_final_cost_by_final_trees_DFS (MST edge + P_D edge)----------------')
                route_cost = calculate_final_cost_by_final_trees(pruned_results, new_edges_set,trees_structure,algorithm.tree_distances, mst_result_trees)

            route_cost_list.append(route_cost)
            
            # end_total_time
            end_total = time.time()

            total_running_time = end_total- start_total

            total_times.append(total_running_time)
            print(f'Final Running time: {total_running_time}')


        x_axle = [f'{num_nodes}_{total_depots}_{level_k}']

        if not is_Greedy:
            all_results[seed_value] = {
                'x_axle': x_axle,
                "CT": CT_times.copy(),                         
                "PD_init": initial_algorithm_times.copy(),     
                "PD_solve": algorithm_times.copy(),            
                "total": total_times.copy(),
                "route_cost": route_cost_list.copy(),
            }
        else:
            all_results[seed_value] = {
                'x_axle': x_axle,
                "total": total_times.copy(),
                "route_cost": route_cost_list.copy(),
                'final_times': final_times.copy()
            }

    if not is_Greedy:
        if constant_MST[0]:
            filename = 'ss_ts_' + str(constant_MST[1])
        else:
            filename = 'ss'  
        output_path = f'{dataset_input}/{routing_method}/{filename}/{dataset_input}_{routing_method}_{filename}_results.csv'
    else:
        output_path = f'{dataset_input}/{routing_method}/greedy_baseline_results.csv'

    save_all_seeds_to_csv(output_path, all_results) 
