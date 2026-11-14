// Dedicated CPU-only inclusion of the actual ContinuousScheduler implementation.
// This separate C++ translation unit MUST never alter the production NVCC
// compilation language/options of distributed/pp2_batch.cu.
#ifndef HERMES_ADMISSION_CPU_ONLY
#error "pp2_batch_cpu_only.cpp must compile with HERMES_ADMISSION_CPU_ONLY"
#endif
#include "../pp2_batch.cu"
