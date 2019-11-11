# Benchutils

* `StatStream` keep tracks of each iteration and compute `average`, `min`, `max`, `sd`, `total`.
* `call_graph` use `pycallgraph` to generate a call graph. The functionality can be disabled using the `NO_CALL_GRAPHS` flag
* `chrono` function decorator to check the runtinme of the function
* `MultiStageChrono` chrono that start a new stage every time start is called a report can be generated later on
* `versioning` retrieve the git commit hash and git commit time to keep track of performance as code evolve
* `report` generate a simple CSV/markdown table from python lists 


# Examples

## StatStream

StatStream is thread safe and multiprocessing friendly (it will sync across processes)

    # Drop the first 5 observations
    
    stat = StatStream(drop_first_obs=5)
    
    for i in range(0, 10):
        start = time.time()
        
        something_long()
        
        stat.update(time.time() - start)
        
        
    stat.avg
    stat.max
    stat.min
    stat.count  # == 5
    stat.sd     
    
# Multi Stage Chrono

    chrono = MultiStageChrono(2)

    for i in range(0, 10):

        with chrono.time('forward_back'):
            with chrono.time('forward'):
                time.sleep(1)

            with chrono.time('backward', skip_obs=3):
                time.sleep(1)
                
        show_eta(i, 10, timer)
        

    chrono.report()
    chrono.report(format='json')
    
Output

    [ 0/10] 22.52 s +/- 0.00 s
    [ 1/10] 18.01 s +/- 2.09 s
    [ 2/10] 16.35 s +/- 1.84 s
    [ 3/10] 13.51 s +/- 1.81 s
    [ 4/10] 11.51 s +/- 1.62 s
    [ 5/10]  9.01 s +/- 1.48 s
    [ 6/10]  6.86 s +/- 1.27 s
    [ 7/10]  4.50 s +/- 1.04 s
    [ 8/10]  2.28 s +/- 0.73 s
    [ 9/10]  0.00 s +/- 0.00 s

            Stage , Average , Deviation ,    Min ,    Max , count 
     forward_back ,  2.0019 ,    0.0003 , 2.0013 , 2.0022 ,     8 
          forward ,  1.0007 ,    0.0003 , 1.0001 , 1.0010 ,     8 
         backward ,  1.0010 ,    0.0000 , 1.0010 , 1.0011 ,     7 

    {
      "forward_back": {
        "avg": 2.0022916555404664,
        "min": 2.0014455318450928,
        "max": 2.003028154373169,
        "sd": 0.000445246188356245,
        "count": 10,
        "unit": "s"
      },
      "forward": {
        "avg": 1.0009073734283447,
        "min": 1.000166893005371,
        "max": 1.0011417865753174,
        "sd": 0.0003713474616615058,
        "count": 10,
        "unit": "s"
      },
      "backward": {
        "avg": 1.0010595662253243,
        "min": 1.0008213520050049,
        "max": 1.0011804103851318,
        "sd": 0.00010404040982473873,
        "count": 7,
        "unit": "s"
      }
    }

    
# Versioning


    import my_module
    
    # return commit hash and commit date
    get_git_version(my_module)
    
    # return sha256 of the given file
    get_file_version(__file__)
    
   
# Chrono


    @chrono
    def my_function():
        pass
        
# Report

    print_table(
        ['A', 'B', 'C'],
        [
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3],
        ]
    )
   