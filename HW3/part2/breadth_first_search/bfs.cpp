#include "bfs.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstddef>
#include <omp.h>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

#define ROOT_NODE_ID 0
#define NOT_VISITED_MARKER -1

void vertex_set_clear(vertex_set *list)
{
    list->count = 0;
}

void vertex_set_init(vertex_set *list, int count)
{
    list->max_vertices = count;
    list->vertices = (int *)malloc(sizeof(int) * list->max_vertices);
    vertex_set_clear(list);
}

// Take one step of "top-down" BFS.  For each vertex on the frontier,
// follow all outgoing edges, and add all neighboring vertices to the
// new_frontier.
void top_down_step(
    Graph g,
    vertex_set *frontier,
    int *distances,
    int iteration)
{
    int local_count = 0;

    // #pragma omp parallel num_threads(NUM_THREADS) private(local_count)
     #pragma omp parallel
    {
        #pragma omp for reduction(+:local_count)
        for (int i = 0; i < g->num_nodes; i++) {
	    if (frontier->vertices[i] == iteration){
		int start_edge = g->outgoing_starts[i];
            	int end_edge = (i == g->num_nodes-1) ? g->num_edges : g->outgoing_starts[i+1];

            	// attempt to add all neighbors to the new frontier
            	for (int neighbor = start_edge; neighbor < end_edge; neighbor++) {
            	    int outgoing = g->outgoing_edges[neighbor];
		    if (frontier->vertices[outgoing] == NOT_VISITED_MARKER){
			local_count++;
			distances[outgoing] = distances[i] + 1;
			frontier->vertices[outgoing] = iteration + 1;
		    }
            	}
	    }
        }
    }

    frontier->count = local_count;

    // for (int i = 0; i < frontier->count; i++)
    // {

    //     int node = frontier->vertices[i];

    //     int start_edge = g->outgoing_starts[node];
    //     int end_edge = (node == g->num_nodes - 1)
    //                        ? g->num_edges
    //                        : g->outgoing_starts[node + 1];

    //     // attempt to add all neighbors to the new frontier
    //     for (int neighbor = start_edge; neighbor < end_edge; neighbor++)
    //     {
    //         int outgoing = g->outgoing_edges[neighbor];

    //         if (distances[outgoing] == NOT_VISITED_MARKER)
    //         {
    //             distances[outgoing] = distances[node] + 1;
    //             int index = new_frontier->count++;
    //             new_frontier->vertices[index] = outgoing;
    //         }
    //     }
    // }
}

// Implements top-down BFS.
//
// Result of execution is that, for each node in the graph, the
// distance to the root is stored in sol.distances.
void bfs_top_down(Graph graph, solution *sol)
{

    vertex_set list1;
    vertex_set_init(&list1, graph->num_nodes);
    int iteration = 0;
    vertex_set *frontier = &list1;

    memset(frontier->vertices, NOT_VISITED_MARKER, sizeof(int) * graph->num_nodes);
    frontier->vertices[ROOT_NODE_ID] = iteration;
    frontier->count++;

    // initialize all nodes to NOT_VISITED
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // setup frontier with the root node
    sol->distances[ROOT_NODE_ID] = 0;

    while (frontier->count != 0)
    {

#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif

        frontier->count = 0;

        top_down_step(graph, frontier, sol->distances, iteration);

#ifdef VERBOSE
        double end_time = CycleTimer::currentSeconds();
        printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

        // swap pointers
	iteration++;
    }
}

void bottom_up_step(
    Graph g,
    vertex_set* frontier,
    int *distances,
    int iteration)
{
    int local_count = 0;

    #pragma omp parallel
    {
	#pragma omp for reduction(+:local_count)
	for (int i=0; i < g->num_nodes; i++) {
    	    if (frontier->vertices[i] == NOT_VISITED_MARKER){
    	        int start_edge = g->incoming_starts[i];
    	        int end_edge = (i == g->num_nodes-1) ? g->num_edges : g->incoming_starts[i+1];

    	        for (int neighbor = start_edge; neighbor < end_edge; neighbor++) {
		    int incoming = g->incoming_edges[neighbor];

		    if (frontier->vertices[incoming] == iteration){
			distances[i] = distances[incoming] + 1;
			frontier->vertices[i] = iteration + 1;
    	    	      	local_count++;
    	    	      	break;
		    }
    	        }
    	    }
    	}
    }
    frontier->count += local_count;
}


void bfs_bottom_up(Graph graph, solution *sol)
{
    // For PP students:
    //
    // You will need to implement the "bottom up" BFS here as
    // described in the handout.
    //
    // As a result of your code's execution, sol.distances should be
    // correctly populated for all nodes in the graph.
    //
    // As was done in the top-down case, you may wish to organize your
    // code by creating subroutine bottom_up_step() that is called in
    // each step of the BFS process.

    vertex_set list1;
    vertex_set_init(&list1, graph->num_nodes);
    int iteration = 0;
    vertex_set *frontier = &list1;

    memset(frontier->vertices, NOT_VISITED_MARKER, sizeof(int) * graph->num_nodes);
    frontier->vertices[ROOT_NODE_ID] = iteration;
    frontier->count++;

    // initialize all nodes to NOT_VISITED
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    sol->distances[ROOT_NODE_ID] = 0;

    while (frontier->count != 0){
        frontier->count = 0;

#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif

        bottom_up_step(graph, frontier, sol->distances, iteration);

#ifdef VERBOSE
        double end_time = CycleTimer::currentSeconds();
        printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

	iteration++;

    }

}

void bfs_hybrid(Graph graph, solution *sol)
{
    // For PP students:
    //
    // You will need to implement the "hybrid" BFS here as
    // described in the handout.


}
