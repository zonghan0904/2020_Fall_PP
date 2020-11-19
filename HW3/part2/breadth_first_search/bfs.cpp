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
#define NOT_FRONTIER_MARKER 0
#define THRESHOLD 10000000

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
    int frontier_id)
{
    int next_frontier_cnt = 0;

     #pragma omp parallel
    {
        #pragma omp for reduction(+:next_frontier_cnt)
        for (int i = 0; i < g->num_nodes; i++) {
	    if (frontier->vertices[i] == frontier_id){
		int start_edge = g->outgoing_starts[i];
            	int end_edge = (i == g->num_nodes-1) ? g->num_edges : g->outgoing_starts[i+1];

            	// attempt to add all neighbors to the new frontier
            	for (int neighbor = start_edge; neighbor < end_edge; neighbor++) {
            	    int neighbor_node = g->outgoing_edges[neighbor];
		    if (frontier->vertices[neighbor_node] == NOT_FRONTIER_MARKER){
			next_frontier_cnt++;
			distances[neighbor_node] = distances[i] + 1;
			frontier->vertices[neighbor_node] = frontier_id + 1;
		    }
            	}
	    }
        }
    }

    frontier->count = next_frontier_cnt;
}

// Implements top-down BFS.
//
// Result of execution is that, for each node in the graph, the
// distance to the root is stored in sol.distances.
void bfs_top_down(Graph graph, solution *sol)
{

    vertex_set list1;
    vertex_set_init(&list1, graph->num_nodes);
    int frontier_id = 1;
    vertex_set *frontier = &list1;

    memset(frontier->vertices, NOT_FRONTIER_MARKER, sizeof(int) * graph->num_nodes);	// initialize to 0 instead of -1 is faster
    frontier->vertices[ROOT_NODE_ID] = frontier_id;
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

        top_down_step(graph, frontier, sol->distances, frontier_id);

#ifdef VERBOSE
        double end_time = CycleTimer::currentSeconds();
        printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

        // swap pointers
	frontier_id++;
    }
}

void bottom_up_step(
    Graph g,
    vertex_set* frontier,
    int *distances,
    int frontier_id)
{
    int next_frontier_cnt = 0;

    #pragma omp parallel
    {
	#pragma omp for reduction(+:next_frontier_cnt)
	for (int i=0; i < g->num_nodes; i++) {
    	    if (frontier->vertices[i] == NOT_FRONTIER_MARKER){
    	        int start_edge = g->incoming_starts[i];
    	        int end_edge = (i == g->num_nodes-1) ? g->num_edges : g->incoming_starts[i+1];

    	        for (int neighbor = start_edge; neighbor < end_edge; neighbor++) {
		    int neighbor_node = g->incoming_edges[neighbor];

		    if (frontier->vertices[neighbor_node] == frontier_id){
			distances[i] = distances[neighbor_node] + 1;
			frontier->vertices[i] = frontier_id + 1;
    	    	      	next_frontier_cnt++;
    	    	      	break;
		    }
    	        }
    	    }
    	}
    }
    frontier->count += next_frontier_cnt;
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
    int frontier_id = 1;
    vertex_set *frontier = &list1;

    memset(frontier->vertices, NOT_FRONTIER_MARKER, sizeof(int) * graph->num_nodes);	// initialize to 0 instead of -1 is faster
    frontier->vertices[ROOT_NODE_ID] = frontier_id;
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

        bottom_up_step(graph, frontier, sol->distances, frontier_id);

#ifdef VERBOSE
        double end_time = CycleTimer::currentSeconds();
        printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

	frontier_id++;

    }

}

void bfs_hybrid(Graph graph, solution *sol)
{
    // For PP students:
    //
    // You will need to implement the "hybrid" BFS here as
    // described in the handout.
    vertex_set list1;
    vertex_set_init(&list1, graph->num_nodes);
    int frontier_id = 1;
    vertex_set *frontier = &list1;

    memset(frontier->vertices, NOT_FRONTIER_MARKER, sizeof(int) * graph->num_nodes);	// initialize to 0 instead of -1 is faster
    frontier->vertices[ROOT_NODE_ID] = frontier_id;
    frontier->count++;

    // initialize all nodes to NOT_VISITED
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    sol->distances[ROOT_NODE_ID] = 0;

    while (frontier->count != 0){
#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif

	if (frontier->count > THRESHOLD){
	    frontier->count = 0;
	    bottom_up_step(graph, frontier, sol->distances, frontier_id);
	}
	else{
	    frontier->count = 0;
	    top_down_step(graph, frontier, sol->distances, frontier_id);
	}

#ifdef VERBOSE
        double end_time = CycleTimer::currentSeconds();
        printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

	frontier_id++;

    }

}
