#pragma once

#include <numa.h>

#ifndef NUMA_NODE_MCDRAM
#define NUMA_NODE_MCDRAM 1
#endif
#define numa_alloc(size) numa_alloc_onnode(size, NUMA_NODE_MCDRAM)
