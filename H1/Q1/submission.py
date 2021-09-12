import http.client
import json
import csv


#############################################################################################################################
# cse6242 
# All instructions, code comments, etc. contained within this notebook are part of the assignment instructions.
# Portions of this file will auto-graded in Gradescope using different sets of parameters / data to ensure that values are not
# hard-coded.
#
# Instructions:  Implement all methods in this file that have a return
# value of 'NotImplemented'. See the documentation within each method for specific details, including
# the expected return value
#
# Helper Functions:
# You are permitted to write additional helper functions/methods or use additional instance variables within
# the `Graph` class or `TMDbAPIUtils` class so long as the originally included methods work as required.
#
# Use:
# The `Graph` class  is used to represent and store the data for the TMDb co-actor network graph.  This class must
# also provide some basic analytics, i.e., number of nodes, edges, and nodes with the highest degree.
#
# The `TMDbAPIUtils` class is used to retrieve Actor/Movie data using themoviedb.org API.  We have provided a few necessary methods
# to test your code w/ the API, e.g.: get_movie_cast(), get_movie_credits_for_person().  You may add additional
# methods and instance variables as desired (see Helper Functions).
#
# The data that you retrieve from the TMDb API is used to build your graph using the Graph class.  After you build your graph using the
# TMDb API data, use the Graph class write_edges_file & write_nodes_file methods to produce the separate nodes and edges
# .csv files for use with the Argo-Lite graph visualization tool.
#
# While building the co-actor graph, you will be required to write code to expand the graph by iterating
# through a portion of the graph nodes and finding similar artists using the TMDb API. We will not grade this code directly
# but will grade the resulting graph data in your Argo-Lite graph snapshot.
#
#############################################################################################################################


class Graph:

    # Do not modify
    def __init__(self, with_nodes_file=None, with_edges_file=None):
        """
        option 1:  init as an empty graph and add nodes
        option 2: init by specifying a path to nodes & edges files
        """
        self.nodes = []
        self.edges = []
        if with_nodes_file and with_edges_file:
            nodes_CSV = csv.reader(open(with_nodes_file))
            nodes_CSV = list(nodes_CSV)[1:]
            self.nodes = [(n[0], n[1]) for n in nodes_CSV]

            edges_CSV = csv.reader(open(with_edges_file))
            edges_CSV = list(edges_CSV)[1:]
            self.edges = [(e[0], e[1]) for e in edges_CSV]


    def add_node(self, id: str, name: str) -> None:
        for node in self.nodes: 
            if node[0] == id:
                print('Node exists')
                return

        self.nodes.append([id, name])

        return
    

    def add_edge(self, source: str, target: str) -> None:
        for edge in self.edges:
            if ((edge[0] == source and edge[1] == target) or (edge[1] == source and edge[0] == target) or (source == target)):
                return

        self.edges.append([source, target])
            
        return


    def total_nodes(self) -> int:
        return len(self.nodes)


    def total_edges(self) -> int:
        """
        Returns an integer value for the total number of edges in the graph
        """
        return len(self.edges)


    def max_degree_nodes(self) -> dict:
        dict = {}

        max_Degree = 0

        for edge in self.edges:
            source = edge[0]
            target = edge[1]

            dict[source] = 1 if source not in dict else dict[source] + 1
            dict[target] = 1 if target not in dict else dict[target] + 1
            max_Degree = max([max_Degree, dict[source], dict[target]])

        filtered_dict =  {k:v for (k,v) in dict.items() if v == max_Degree }

        return filtered_dict


    def print_nodes(self):
        """
        No further implementation required
        May be used for de-bugging if necessary
        """
        print(self.nodes)


    def print_edges(self):
        """
        No further implementation required
        May be used for de-bugging if necessary
        """
        print(self.edges)


    # Do not modify
    def write_edges_file(self, path="edges.csv")->None:
        """
        write all edges out as .csv
        :param path: string
        :return: None
        """
        edges_path = path
        edges_file = open(edges_path, 'w', encoding='utf-8')

        edges_file.write("source" + "," + "target" + "\n")

        for e in self.edges:
            edges_file.write(e[0] + "," + e[1] + "\n")

        edges_file.close()
        print("finished writing edges to csv")


    # Do not modify
    def write_nodes_file(self, path="nodes.csv")->None:
        """
        write all nodes out as .csv
        :param path: string
        :return: None
        """
        nodes_path = path
        nodes_file = open(nodes_path, 'w', encoding='utf-8')

        nodes_file.write("id,name" + "\n")
        for n in self.nodes:
            nodes_file.write(n[0] + "," + n[1] + "\n")
        nodes_file.close()
        print("finished writing nodes to csv")



class  TMDBAPIUtils:

    # Do not modify
    def __init__(self, api_key:str):
        self.api_key=api_key


    def get_movie_cast(self, movie_id:str, limit:int=None, exclude_ids:list=None) -> list:
        conn = http.client.HTTPSConnection("api.themoviedb.org")
        conn.request("GET", "/3/movie/%s/credits?api_key=%s&language=en-US" % (movie_id, self.api_key))
        response = conn.getresponse()
        if response.status == 200:
            credits_json = response.read().decode('utf-8')
            credits = json.loads(credits_json)
            casts = credits['cast']
            if  exclude_ids is not None: 
                casts = filter(lambda cast: cast['id'] not in exclude_ids, casts)
            if limit is not None:
                casts = filter(lambda cast: cast['order'] < limit, casts)
        return list(casts)


    def get_movie_credits_for_person(self, person_id:str, vote_avg_threshold:float=None)->list:
        """
        Using the TMDb API, get the movie credits for a person serving in a cast role
        documentation url: https://developers.themoviedb.org/3/people/get-person-movie-credits

        :param string person_id: the id of a person
        :param vote_avg_threshold: optional parameter to return the movie credit if it is >=
            the specified threshold.
            e.g., if the vote_avg_threshold is 5.0, then only return credits with a vote_avg >= 5.0
        :rtype: list
            return a list of dicts, one dict per movie credit with the following structure:
                [{'id': '97909' # the id of the movie credit
                'title': 'Long, Stock and Two Smoking Barrels' # the title (not original title) of the credit
                'vote_avg': 5.0 # the float value of the vote average value for the credit}, ... ]
        """
        conn = http.client.HTTPSConnection("api.themoviedb.org")
        conn.request("GET", "/3/person/%s/movie_credits?api_key=%s&language=en-US"%(person_id, self.api_key))
        response = conn.getresponse()
        if response.status == 200:
            movie_credits_response = response.read().decode('utf-8')
            movie_credits = json.loads(movie_credits_response)
            casts = movie_credits['cast']
            if  vote_avg_threshold is not None: 
                casts = filter(lambda cast: cast['vote_average'] >= vote_avg_threshold, casts)
        return list(casts)

# Exception handling and best practices
# - You should use the param 'language=en-US' in all API calls to avoid encoding issues when writing data to file.
# - If the actor name has a comma char ',' it should be removed to prevent extra columns from being inserted into the .csv file
# - Some movie_credits may actually be collections and do not return cast data. Handle this situation by skipping these instances.
# - While The TMDb API does not have a rate-limiting scheme in place, consider that making hundreds / thousands of calls
#   can occasionally result in timeout errors. If you continue to experience 'ConnectionRefusedError : [Errno 61] Connection refused',
#   - wait a while and then try again.  It may be necessary to insert periodic sleeps when you are building your graph.


def return_name()->str:
    return "agjoka3"


def return_argo_lite_snapshot()->str:
    return 'https://poloclub.github.io/argo-graph-lite/#9f201e04-b2c1-4cb7-804d-c495eae5a1fc'



# You should modify __main__ as you see fit to build/test your graph using  the TMDBAPIUtils & Graph classes.
# Some boilerplate/sample code is provided for demonstration. We will not call __main__ during grading.

if __name__ == "__main__":

    graph = Graph()
    graph.add_node(id='2975', name='Laurence Fishburne')
   
    tmdb_api_utils = TMDBAPIUtils(api_key='92abb2bbcb4ae1f3a26d7229ccf285f0')

    # call functions or place code here to build graph (graph building code not graded)
    # Suggestion: code should contain steps outlined above in BUILD CO-ACTOR NETWORK

    movies = tmdb_api_utils.get_movie_credits_for_person('2975', 8)
    for movie in movies:
        cast_members = tmdb_api_utils.get_movie_cast(movie['id'], 3)
        for member in cast_members:
            graph.add_node(str(member['id']), member['name'])
            graph.add_edge( '2975', str(member['id']))

    i = 0
    first_iteration_nodes_ids = []
    while i < 2:
        nodes = graph.nodes
        print(graph.total_nodes())
        i += 1
        for node in nodes:
            movies_credits = tmdb_api_utils.get_movie_credits_for_person(node[0], 8)
            for movie in movies_credits:
                if graph.total_nodes() > 1000:
                    break
                exclude_ids = []
                if i == 0:
                    exclude_ids = ['2975']
                else:
                    exclude_ids = first_iteration_nodes_ids
                cast_members = tmdb_api_utils.get_movie_cast(movie['id'], 3, exclude_ids)
                for member in cast_members:
                    graph.add_node(str(member['id']), member['name'])
                    if i == 0:
                        first_iteration_nodes_ids.append(str(member['id']))
                    print(graph.total_nodes())
                    graph.add_edge(node[0], str(member['id']))
    
    graph.write_edges_file()
    graph.write_nodes_file()


    # If you have already built & written out your graph, you could read in your nodes & edges files
    # to perform testing on your graph.
    # graph = Graph(with_edges_file="edges.csv", with_nodes_file="nodes.csv")