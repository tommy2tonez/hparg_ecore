//this project probably takes at least 20 MB of raw code - this is a little overwhelming tbh
//I hope that I could get this done within a year or so - to be realistic 
//a first cut implementation probably done within a month or two

//MVPs (engine-wise):   - 1% gpu_ram_size/ host_disk_size
//                      - 90% gpu utility at all time
//                      - able to saturate 95% of network bandwidth at all time

//MVPs (product-wise (so called Large Language Model)):  - uniform distribution of logit influence - given all pair of logits, an influence of a logit on another is unif dist among all pairs (after certain transformation hops)
//                                                       - 0.001% - 0.01% dimensional reduction rate (cublas linear is a form of reduction + expansion) | 10000 tiles: 1 tile
//                                                       - model-less model (auto differentiation based on path)
//                                                       - inter-communicate networks (think humans)