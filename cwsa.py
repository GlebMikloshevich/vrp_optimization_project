import numpy as np
import matplotlib.pyplot as plt
from VRPGraph import VRPGraph


class CWSAlgorithm:
    """
    Clark-Wright Savings Algorithm
    """
    def __init__(self, vrp_graph: VRPGraph, capacity=1):
        self.vrp_graph = vrp_graph
        self.route = np.random.permutation(self.vrp_graph.num_nodes)
        self.capacity = capacity

    def calc_savings(self):
        savings = []
        for i in range(1, self.vrp_graph.num_nodes):
            for j in range(i, self.vrp_graph.num_nodes):
                if i == j and i != 0:
                    continue
                max_ = max(i, j)
                min_ = min(i, j)
                savings.append(((min_, max_),
                                # use distance to warehouse and from
                                self.vrp_graph.euclid_distance(0, i) + self.vrp_graph.euclid_distance(0, j)
                                - self.vrp_graph.euclid_distance(i, j))
                               )
        savings = sorted(savings, key=lambda x: x[1], reverse=True)[::2]
        return savings

    def run(self):
        routes = []
        assigned_nodes = {i: None for i in range(1, self.vrp_graph.num_nodes)}
        savings = self.calc_savings()

        route_index = 0
        for node_link in savings:
            if assigned_nodes[node_link[0][0]] is None and assigned_nodes[node_link[0][1]] is None:
                if self.vrp_graph.demand[node_link[0][0]] + self.vrp_graph.demand[node_link[0][1]] > self.capacity:
                    continue
                route = [node_link[0][0], node_link[0][1]]
                demand = self._calc_route_demand(route)
                for k in route:
                    assigned_nodes[k] = route_index
                routes.append([route, demand])
                route_index += 1
            # merge route and point
            elif ((assigned_nodes[node_link[0][0]] is None) + (assigned_nodes[node_link[0][1]] is None)) == 1:
                if assigned_nodes[node_link[0][0]] is None:
                    none_index = node_link[0][0]
                    not_none_index = node_link[0][1]
                else:
                    none_index = node_link[0][1]
                    not_none_index = node_link[0][0]
                index = assigned_nodes[not_none_index]
                if self.vrp_graph.demand[none_index] + routes[assigned_nodes[not_none_index]][1] > self.capacity:
                    continue
                if routes[index][0][0] == not_none_index:
                    routes[index][0].insert(0, none_index)
                elif routes[index][0][-1] == not_none_index:
                    routes[index][0].append(none_index)
                else:
                    continue
                assigned_nodes[none_index] = index
                routes[index][1] = self._calc_route_demand(routes[index][0])
            # merge two routes
            elif assigned_nodes[node_link[0][0]] is not None and assigned_nodes[node_link[0][1]] is not None:
                index1 = node_link[0][0]
                index2 = node_link[0][1]
                route1 = assigned_nodes[index1]
                route2 = assigned_nodes[index2]
                if route1 == route2:
                    continue
                if routes[route1][1] + routes[route2][1] > self.capacity:
                    continue
                if routes[route1][0][0] == index1:
                    if routes[route2][0][0] == index2:
                        routes[route1][0] = routes[route1][0][::-1]
                        for node in routes[route2][0]:
                            routes[route1][0].append(node)
                    elif routes[route2][0][-1] == index2:
                        routes[route1][0] = routes[route1][0][::-1]
                        for node in routes[route2][0][::-1]:
                            routes[route1][0].append(node)
                elif routes[route1][0][-1] == index1:
                    if routes[route2][0][0] == index2:
                        for node in routes[route2][0]:
                            routes[route1][0].append(node)
                    elif routes[route2][0][-1] == index2:
                        for node in routes[route2][0][::-1]:
                            routes[route1][0].append(node)
                for node in routes[route1][0]:
                    assigned_nodes[node] = route1
                routes[route1][1] = self._calc_route_demand(routes[route1][0])

        used_routes = [False for _ in range(len(routes))]
        cleaned_routes = []
        for v in assigned_nodes.values():
            if v is None:
                continue
            if not used_routes[v]:
                cleaned_routes.append(routes[v])
                used_routes[v] = True
        return cleaned_routes

    def update_vrp(self, vrp_graph: VRPGraph):
        self.vrp_graph = vrp_graph

    def _calc_route_demand(self, route):
        demand = 0
        for vertex in route:
            demand += self.vrp_graph.demand[vertex]
        return demand

    def generate_alikas_route(self):
        routes = self.run()
        alikas_route = []
        for route in routes:
            alikas_route.append(0)
            for vertex in route[0]:
                alikas_route.append(vertex)
            alikas_route.append(0)
        return alikas_route


if __name__ == "__main__":
    vrp_graph = VRPGraph(num_nodes=20, num_depots=2, plot_demand=True)  # Adjust num_depots if necessary
    cwsa = CWSAlgorithm(vrp_graph)
    print(cwsa.generate_alikas_route())
