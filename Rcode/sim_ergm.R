library(ergm)
library(network)


# construct the networks
construct_er_model = function(d){
  un=network(d, directed = FALSE)
  model0<- un ~ edges
}


construct_e2s_model = function(d){
  un=network(d, directed = FALSE)
  model0<- un ~ edges +kstar(2)
}


construct_e2st_model = function(d){
  un=network(d, directed = FALSE)
  model0<- un ~ edges + kstar(2) + triangles
}



# generate the network adjacency matrices  


gen_ergm = function(d=20, N=500, construct=construct_er_model, coef=c(0)){
    # d: size of the network
    # N: number of network samples
    # construct: the method to construct ergm model
    #coef: coeficient for network statistics
    model0 = construct(d)
    g.sim  <- simulate(model0, nsim=N, coef=coef, 
                        control=control.simulate(MCMC.burnin=100+10*d, MCMC.interval=d))
    
    if (N==1){
        g.adj = g.sim[,]}
    else{
        g.adj = c()
        for (ii in 1:N){
                g.adj= c(g.adj, list(g.sim[[ii]][,]))
            }
    }
    # list(g.sim, g.adj)
    g.adj
}

