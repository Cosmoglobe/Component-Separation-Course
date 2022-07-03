
def split_workload(N, mpi_rank, mpi_size):
    """
    Method for equally splitting the workload between processors 

    Returns
    -------
    chunk_start int 
    chunk_end   int
    """
    # Splitting it equally
    workloads = [ N // mpi_size for i in range(mpi_size) ]
    # Adding unequal part
    for i in range( N % mpi_size ):
        workloads[i] += 1

    chunk_start = 0
    for i in range( mpi_rank ):
        chunk_start += workloads[i]
    chunk_end = chunk_start + workloads[mpi_rank]
    
    return chunk_start, chunk_end, workloads
