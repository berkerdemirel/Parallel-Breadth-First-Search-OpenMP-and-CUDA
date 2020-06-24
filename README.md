# Direction-Optimizing-BFS

Implementation of Direction Optimizing BFS in both OpenMP and CUDA (C++)

WBFS -worker breadth first search- scans the entire set of vertices exhaustively at each iteration to determine the current frontier
QBFS -queue breadth first search- maintains a global frontier and generates the next frontier by using the current one.
HYBRID combines top-down and bottom-up approaches to optimize the search direction. When frontier gets too large, it is easier to start from unvisited nodes to visited, when searching.

## WBFS

WBFS does not require any synchronization and it does not suffer from queue overhead. However it is expected for WBFS to perform poorly when it encounters a big graph with small diameter since it traverse each vertex at each level (for small gain).

## QBFS

QBFS improves WBFS's redundant iterations by using a queue. However it faces a synchronization overhead.
The main trick to reduce the synchronization overhead is to use local frontiers for each thread to accumulate the next state of the global frontier. 
At the end of each step, the local frontiers are merged into the global one by a single thread. This is achieved with performing a merge operation by maintaining a range of indices for each thread by computing a prefix sum so that they can write to the global frontier without any conflicts.

This approach saves us from pushing each new vertex to the frontier atomically, hence decreases the total cost of synchronization.


## HYBRID

HYBRID approach is a direction optimizing approach, in top down part (when frontier is relatively small), QBFS is used. As time goes frontier gets larger and larger. Then algorithm switches to bottom up approach to make use of large frontier by starting from unvisited and searching for visited nodes.

The algorithm keeps track of number of visited and unvisited nodes to decide when to switch from one to other.



According to experimental results, use QBFS when the average degree is small, otherwise use HYBRID version.


## References

[1] The Matrix Market File Format. http://networkrepository.com/mtx-matrix-market-format.html.

[2] Scott Beamer, Krste Asanovic, and David Patterson. Direction-optimizing breadth-first search. In Proceedings of the International Conference on High Performance Computing, Networking, Storage and Analysis, SC ’12, pages 12:1–12:10, Los Alamitos, CA, USA, 2012. IEEE Computer Society Press.

[3] Duane Merrill, Michael Garland, and Andrew Grimshaw. Scalable gpu graph traversal. In Proceedings of the 17th ACM SIGPLAN Symposium on Principles and Practice of Parallel Programming, PPoPP ’12, pages 117–128, New York, NY, USA, 2012. ACM.

[4] Julian Shun and Guy E. Blelloch. Ligra: A lightweight graph processing framework for shared memory. SIGPLAN Not., 48(8):135–146, February 2013.

[5] Yangzihao Wang, Andrew Davidson, Yuechao Pan, YuduoWu, Andy Riffel, and John D. Owens. Gunrock: A high-performance graph processing library on the gpu. SIGPLAN Not., 51(8):11:1–11:12, February 2016.

