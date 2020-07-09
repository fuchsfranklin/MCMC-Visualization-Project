library(shiny)
library(shinythemes)
library(ggplot2)
library(gganimate)
library(shinycustomloader)
library(transformr)
library(tidyr)
library(gifski)
theme_set(theme_bw())

#UI
###################################################################
ui <- navbarPage(theme = shinytheme("flatly"),
                 
                 title = "Metropolis-Hastings MCMC",
                 
#Pages                 
###################################################################                 
                 tabPanel(withMathJax(),
                          title = "Introduction and Background",
                          h1("Introduction and Background"),
                          hr(),
                          h4("A Stochastic Models and Simulation Project"),
                          h4("Franklin Fuchs"),
                          h6("University of Nevada, Reno"),
                          hr(),
                          h2("Introduction"),
                          withMathJax(sprintf("The aim of this stochastic models and simulation project is to understand the metropolis-hastings algorithm and several markov-chain-monte-carlo diagnostic methods at a more intuitive and visual level through plots that are both animated and interactive. My second aim is to present my first aim in a cohesive and compect manner to those unfamiliar with MCMC and the R programming language. It is important to mention that the time-dependent nature of a markov chain and the amount of visually appealing parameters are optimal for creating animated illustrations. This project was developed as a Web-Application using R-Shiny. The next step I will take for this project is to add a tab that brings all other concepts together in one cohesive analysis.")),
                          h2("Markov Chains and MCMC"),
                          withMathJax(sprintf("The first step in understanding the metropolis-hastings algorithm is understanding the notion of a markov chain, which is a sequence a of states where the probability of transitioning from states only depends on the current state. A markov chain can be formally defined as sequence of random variables \\(\\{X_t\\}\\) on state space \\(\\Omega\\) for \\(t=0,1,2,...\\), such that $$P(X_{t+1}=y|X_{t}=x_{t},...,X_{0}=x_{0})$$ $$\\qquad \\quad =P(X_{t+1}=y|X_{t}=x_{t})$$ $$\\qquad \\qquad \\quad \\implies P(X_{t+1}=i|X_{t}=j)=p_{ij}$$ Where \\(P=(p_{ij})\\) is the transition probability matrix with dimensions \\(|\\Omega |\\times|\\Omega |\\). Intuitively, the aforementioned transition probability matrix can be understood to contain the probabilities for a transition from one state to another for all states.  The next step in understanding the metropolis-hastings algorithm concerns markov-chain-monte-carlo (MCMC) methods, which we will consider in a big-picture manner with some practical examples. The MCMC class of algorithms are used for sampling from a probability distribution where a Markov Chain that has the desired distribution as its stationary distribution is constructed. The random samples are generated in a sequential process where each sample is only dependent on the one before it, as we know from considering markov chain properties earlier. In general, MCMC methods are useful for approximating solutions to problems which cannot be solved analytically or problems regarding optimisation. To understand where MCMC methods can be applied we consider the the following overview outlined by Andrieu et al. (2003).")),
                          tags$br(),
                          tags$br(),
                          tags$ol(
                            tags$li(withMathJax(tags$b("Bayesian Inference and Learning: "),
                                                tags$br(),
                                                sprintf("Given unknown variables \\(x\\in X\\) and data \\(y\\in Y\\), integration problems of normalisation, marginalisation and evaluating expectation can be numerically adressed with MCMC. As an example, consider obtaining the posterior \\(p(x|y)\\) given prior \\(p(x)\\) and likelihood \\(p(y|x)\\), the denominator of bayes theorem needs to be numerically computed as $$p(x|y)=\\frac{p(y|x)p(x)}{\\int_X p(y|x')p(x')dx'}$$"))),
                            tags$li(withMathJax(tags$b("Statistical Mechanics: "),
                                                tags$br(),
                                                sprintf("As an example, let us consider the following. Given states \\(s\\), sum of potential and kinetic energies for all particles in a system \\(E(s)\\), temperature \\(T\\), and boltzmann's constant \\(k\\), the partition function (to describe statistical properties) of a system Z can be calculated as $$Z=\\sum_{s}exp[-\\frac{E(s)}{kT}]$$"))),
                            tags$li(withMathJax(tags$b("Optimisation: "),
                                                tags$br(),
                                                sprintf("The role of MCMC in optimisation is to find a solution to minimize some objective function from a set of given solutions which is often too computationally expensive to be completely explored. As an example that cannot be solved analytically, consider function \\(f(x,y)=z\\) where we want to find optimal values of \\(x'\\) and \\(y'\\) such that $$(x_1 ,...,x_n) \\text{ and } (y_1 ,...,y_n) \\geq f(x,y) \\forall (x,y),x \\in X,y \\in Y $$"))),
                            tags$li(withMathJax(tags$b("Penalised Likelihood Model Selection: "),
                                                tags$br(),
                                                sprintf("Concerning penalised model selection, maximum likelihood estimates are first found for each model seperately and then one model is selected via selection term such as \\(AIC\\) or \\(BIC\\), where the set of initial models can be very large.")))
                          ),
                          tags$br(),
                          h2("Metropolis-Hastings MCMC"),
                          withMathJax(sprintf("Now that we have defined markov chains and considered applications of MCMC, we can finally consider the metropolis-hastings (MH) algorithm. We use the MH-algorithm to generate a sequence of random samples, which can be used to approximate a distribution directly or a distribution parameter of a high-dimensional distribution, where sampling is difficult, through integration. An important property of the MH-Algorithm is that we can draw samples from any probability distribution \\(P(x)\\) provided that can compute the probability density function of \\(g(x)\\) such that \\(g(x) \\propto P(x) \\). Let us now consider the actual algorithm, for which we define target distribution \\(p(x)\\), proposal distribution \\(q(x^*|x)\\) and acceptance probability \\(A(x,x^*)\\), where we have the algorithm as")),
                          tags$br(),
                          hr(),
                          h4("MH Algorithm"),
                          tags$ol(
                            tags$li(withMathJax(sprintf("Initialise \\(x_0\\)"))),
                            tags$li(withMathJax(sprintf("For \\(i=0\\)  to  \\(N-1\\)")),
                                    tags$ul(
                                      tags$li(withMathJax(sprintf("Sample \\(u \\sim U(0,1)\\)."))),
                                      tags$li(withMathJax(sprintf("Sample \\(x^* \\sim q(x^* ,x^{(i)})\\)."))), 
                                      tags$li(withMathJax(sprintf("If \\(u < A(x^{(i)},x^*)=min\\{1,\\frac{p(x^{*}q(x^{i}|x*)}{p(x^{(i)}q(x^*|x^{(i)}))}\\}\\) $$x^{(i+1)}=x^*$$ else $$x^{(i+1)}=x^{i}$$")))
                                    )
                            )
                          ),
                          hr(),
                          tags$br(),
                          p("Considering the algorithm above, observe that it generates a random walk using a proposal distribution which can be used as a sample from which statistics corresponding to the target distribution can be computed from. To view the animated algorithm on several simple distributions, view the animated sampling tab. To conclude this introduction, it is important to mention that real-world target distributions are more complicated, harder to compute, and often without a normalizing constant than the illustrative examples that are used in these illustrative examples.")
                 ),

################################################################### 
                 navbarMenu(title = "Animated Sampling",
                            
                          tabPanel(title = "Uniform Proposal & Normal Target",
                          h1("Uniform Proposal & Normal Target"),
                          tags$ul(
                            tags$li(withMathJax(sprintf("Enter parameters for \\(U_p\\sim (-\\alpha_p,\\alpha_p)\\) and \\(N_t\\sim (\\mu_t,\\sigma_t)\\)."))),
                            tags$li("To simualate a chain that mixes well, start the animation with unchanged default parameters."),
                            tags$li("Animations might take a while to render.")
                          ),
                          sidebarLayout(
                            sidebarPanel(
                              tags$h3("Sampling Parameters"),
                              sliderInput(inputId="iter1",
                                          label=withMathJax(sprintf("Number of Iterations \\(i\\)")),
                                          value=5000,min=100,max=10000),
                              numericInput(inputId="x1_initial",
                                           label=withMathJax(sprintf("Initial Value \\(x_0\\)")),
                                           value=0),
                              numericInput(inputId="p1alpha",
                                                      label=withMathJax(sprintf("Proposal \\(\\alpha\\)")),
                                                      value=1),
                              fluidRow(
                                column(6,numericInput(inputId="t1mu",
                                                      label=withMathJax(sprintf("Target \\(\\mu\\)")),
                                                      value=0)),
                                column(6,numericInput(inputId="t1sdev",
                                                      label=withMathJax(sprintf("Target \\(\\sigma\\)")),
                                                      value=1))
                              ),
                              actionButton(inputId="go1",
                                           label="Start Animation"),
                            ), #endsidebarpanel
                            mainPanel(
                              column(4,withLoader(imageOutput("hist1"),type="html",loader="loader6")),
                              column(8,imageOutput("plot1")),
                            )#end mainpanel
                          )# end sidebarlayout
                       ),
                       
                       tabPanel(title = "Normal Proposal & Normal Target",
                                h1("Normal Proposal & Normal Target"),
                                tags$ul(
                                  tags$li(withMathJax(sprintf("Enter parameters for \\(N_p\\sim (\\mu_p,\\sigma_p)\\) and \\(N_t\\sim (\\mu_t,\\sigma_t)\\)."))),
                                  tags$li("To simulate a chain that mixes well, start the animation with unchanged default parameters."),
                                  tags$li("Animations might take a while to render.")
                                ),
                                sidebarLayout(
                                  sidebarPanel(
                                    tags$h3("Sampling Parameters"),
                                    sliderInput(inputId="iter2",
                                                label=withMathJax(sprintf("Number of Iterations \\(i\\)")),
                                                value=5000,min=100,max=10000),
                                    numericInput(inputId="x2_initial",
                                                 label=withMathJax(sprintf("Initial Value \\(x_0\\)")),
                                                 value=0),
                                    fluidRow(
                                      column(6,numericInput(inputId="p2mu",
                                                            label=withMathJax(sprintf("Proposal \\(\\mu\\)")),
                                                            value=0)),
                                      column(6,numericInput(inputId="p2sdev",
                                                            label=withMathJax(sprintf("Proposal \\(\\sigma\\)")),
                                                            value=1))
                                    ),
                                    fluidRow(
                                      column(6,numericInput(inputId="t2mu",
                                                            label=withMathJax(sprintf("Target \\(\\mu\\)")),
                                                            value=0)),
                                      column(6,numericInput(inputId="t2sdev",
                                                            label=withMathJax(sprintf("Target \\(\\sigma\\)")),
                                                            value=1))
                                    ),
                                    actionButton(inputId="go2",
                                                 label="Start Animation"),
                                  ), #endsidebarpanel
                                  mainPanel(
                                    column(4,withLoader(imageOutput("hist2"),type="html",loader="loader6")),
                                    column(8,imageOutput("plot2")),
                                  )#end mainpanel
                                )# end sidebarlayout
                       ),
                       
                       tabPanel(title = "Normal Proposal & Gamma Target",
                                h1("Normal Proposal & Gamma Target"),
                                tags$ul(
                                  tags$li(withMathJax(sprintf("Enter parameters for \\(N_p\\sim (\\mu_p,\\sigma_p)\\) and \\(\\Gamma_t\\sim (\\mu_t,\\sigma_t)\\)."))),
                                  tags$li("To simulate a chain that mixes well, start the animation with unchanged default parameters."),
                                  tags$li("Animations might take a while to render.")
                                ),
                                sidebarLayout(
                                  sidebarPanel(
                                    tags$h3("Sampling Parameters"),
                                    sliderInput(inputId="iter3",
                                                label=withMathJax(sprintf("Number of Iterations \\(i\\)")),
                                                value=5000,min=100,max=10000),
                                    fluidRow(
                                      column(6,numericInput(inputId="p3mu",
                                                            label=withMathJax(sprintf("Proposal \\(\\mu\\)")),
                                                            value=0)),
                                      column(6,numericInput(inputId="p3sdev",
                                                            label=withMathJax(sprintf("Proposal \\(\\sigma\\)")),
                                                            value=1))
                                    ),
                                    fluidRow(
                                      column(6,numericInput(inputId="t3a",
                                                            label=withMathJax(sprintf("Target \\(a\\)")),
                                                            value=2.3)),
                                      column(6,numericInput(inputId="t3b",
                                                            label=withMathJax(sprintf("Target \\(b\\)")),
                                                            value=2.7))
                                    ),
                                    actionButton(inputId="go3",
                                                 label="Start Animation"),
                                  ), #endsidebarpanel
                                  mainPanel(
                                    column(4,withLoader(imageOutput("hist3"),type="html",loader="loader6")),
                                    column(8,imageOutput("plot3")),
                                  )#end mainpanel
                                )# end sidebarlayout
                       )
                 ),

###################################################################
                 tabPanel(title = "Burning-In",
                          h1("Initial Values"),
                          withMathJax(sprintf("A markov chain usually needs time to reach its equilibrium distribution as we can see by considering the leftmost animation below. This observation makes sense intuitively since there is an infinite number of choices for the initial value \\(x_0\\). Furthermore, different choices of starting points \\(x_0\\) may result in the over-sampling of regions that actually have a low probability under the equilibrium distribution, since some choices of \\(x_0\\) are better than others. The animations below illustrate that that there are clearly better initial value choices for a target distribution that has mean zero, which are choices closer to the true mean zero in this case. Both Chains clearly converge as number of iterations \\(i\\rightarrow \\infty\\), although the chain with the sub-optimal initial value clearly takes longer to converge than the chain with the optimal initial value. This brings us to the idea of burn-in and its removal, as outlined in the following section.")),
                          p(),
                          hr(),
                          fluidRow(
                            column(4,withLoader(imageOutput("plot4"),type="html",loader="loader6")),
                            column(4,withLoader(imageOutput("plot5"),type="html",loader="loader6")),
                            column(4,withLoader(imageOutput("plot6"),type="html",loader="loader6"))
                          ),
                          hr(),
                          tags$br(),
                          tags$br(),
                          h1("Burning-In"),
                          withMathJax(sprintf("Burning-in a markov chain refers to removing the first \\(n\\) samples before starting to record measurements to compute summary statistics from. Although there is no theoretical justification behind burning-in, the practical idea is to give a chain time to converge to the region that has a high probability under the equilibrium distribution and removing the samples where the chain was not yet converged for more accurate inferences. Now consider an animation example that shows how effective burning-in can be with thirty chains at various initial values. Running the animation with no burn-in removal such as \\(n=0\\) and then with a large amount of burn-in removal such as \\(n=500\\) for a large amount of chains empasizes that discarding the first few iterations can a much more accurate idea of the value the chain converges to, especially for chains with sub-optimal initial values, which is especially useful when the aim is to make inferences based on sample statistics of the chain. You can convince yourself how effective burning-in can be by trying different values of \\(n\\) for the simulation below.")),
                          hr(),
                          tags$ul(
                            tags$li(withMathJax(sprintf("To see the effect of burning-in, run simulation with \\(n=0\\) and then \\(n=500\\)"))),
                            tags$li("Animation might take a while to render.")
                          ),
                          sidebarLayout(
                            sidebarPanel(
                              sliderInput(inputId="iterb1",
                                          label=withMathJax(sprintf("Number of first \\(n\\) samples to discard")),
                                          value=0,min=0,max=999), 
                              actionButton(inputId="go5",
                                           label="Start Animation")
                            ),
                            mainPanel(
                              withLoader(imageOutput("plot7"),type="html",loader="loader6") 
                            )
                          ),
                          hr(),
                          h1("A Word of Caution"),
                          withMathJax(sprintf("As mentioned before, there is no theoretical motivation behind removing burn-in. In fact, the underlying theory such as the strong law of large numbers and the central limit theorem hold regardless of the distribution of the initial position \\(x_0\\), as emphasized by Meyn and Tweedie (1993). Geyer takes this idea of burning-in lacking mathematical foundations further by emphasizing that convergence is guaranteed anayway if a run is long enough, and so burning-in should not be employed as a technique. Additionally, Geyer recommends alternatives such as choosing an initial value where the last run ended or choosing a point, such as the mode, which is known to have a reasonably high probality. It is important to mention that these alternatives are not without flaw, given for example a first chain (in a set of chains) with no known high probability point gives no suggested initial value. Thus, the removal of burn-in seems controversial in theory, which is important to keep in mind when burning-in chains."))
                 ),

###################################################################
                 tabPanel(title = "Thinning",
                          h1("Autocorrelation"),
                          withMathJax(sprintf("Relative to other traditional simulation methods, MCMC techniques and the MH-Algortithm are relatively flexible in terms of being able to sample from complex target distributions. Another major difference between traditional simulation methods and MCMC methods is that a MCMC sample is almost always dependent and followingly autocorrelated. Formally, for a sequence of random variables \\(\\{X_{n}\\}\\), the autocorrelation coefficient between terms \\(\\{X_{n}\\}\\) and \\(\\{X_{n+k}\\}\\) is $$\\rho (n,n+k)=\\frac{Cov(X_{n},X_{n+k})}{\\sqrt{Var(X_{n})Var(X_{n+k})}}$$ which can be recognized as the linear correlation coefficient. It is important to emphasize that the coefficient \\(k\\) in the previous equation is the distance between two terms and is abbreviated as lag. We can visualize autocorrelations of a sample with the autocorrelation function (ACF), which maps lags to autocorrelations where \\(\\rho_{k}\\) is a function of \\(k\\). Below are ACF plots of slowly and rapidly decaying autocorrelations for two different samples.")),
                          tags$br(),
                          hr(),
                          fluidRow(
                            column(5, plotOutput("static1")),
                            column(5, plotOutput("static2"),offset=1)
                          ),
                          p("Generating a strongly correlated sample can be problematic when the goal is to simulate an uncorrelated sample from an underlying distribution. A process called thinning, which we explore in the next section, can achieve approximate zero correlation of a sample and independence if the sample is normally distributed."),
                          hr(),
                          h1("Thinning"),
                          withMathJax(sprintf("Thinning refers to the process of keeping only every \\(k_{th}\\) value and discarding all other sampled values. The main benefit of only keeping samples that are spaced out results in an approximate correlation of zero for of the remaining sample, or independence if the sample is normal. Another minor benefit of thinning as outlined by Link and Eaton (2012) of chains is the reduced memory requirement, although storage requirements are not much of a problem in the current age of cheaply available memory. To illustrate the effectiveness of thinning on autocorrelation, we simulate a chain of \\(n=10,000\\) below. After the sample is generated, choose a value for the thinning parameter \\(k\\). Then it can be observed that increasing the value of \\(k\\) can be very effective in reducing autocorrelation for a sample generated by the MH-Algorithm, although there are major drawbacks which are outlined in the next section.")),
                          tags$br(),
                          hr(),
                          tags$ul(
                            tags$li(withMathJax(sprintf("A value of \\(k=1\\) plots the ACF for the the sample without thinning."))),
                            tags$li(withMathJax(sprintf("To see autocorrelation reduction through thinning increase value of \\(k\\).")))
                          ),
                          sidebarLayout(
                            sidebarPanel(
                              sliderInput(inputId="itert1",
                                          label=withMathJax(sprintf("Number of every \\(k_{th}\\) sample value to be kept")),
                                      value=1,min=1,max=10), 
                            ),
                            mainPanel(
                              plotOutput("interactive1")
                            )
                          ),
                          hr(),
                          h1("A Word of Caution"),
                          withMathJax(sprintf("The main drawback of thinning as emphasized by Owen (2017) is the loss of precision that is associated with discarding data. If the goal of an analysis is to maximize precision, decreasing the sample size would be counterintuitive. Furthermore, generating mutliple independent chains and considering sample statistics computed among the values of the indepedent chains produces much better results than thinning. Therefore, the tradeoff between precision and an approximate correlation of zero needs to be kept in mind when considering the use of thinning."))
                 ),

###################################################################
                 tabPanel(title = "References",
                          h2("References"),
                          h4("Papers"),
                          tags$ul(
                            tags$li("Andrieu, C. et al. (2003),An Introduction to MCMC for Machine Learning. Machine Learning, 50: 5-43. https://doi.org/10.1023/A:1020281327116"),
                            tags$li("Link, W.A. and Eaton, M.J. (2012), On thinning of chains in MCMC. Methods in Ecology and Evolution, 3: 112-115. https://doi.org/10.1111/j.2041-210X.2011.00131.x"),
                            tags$li("Owen, Art B. (2017), Statistically efficient thinning of a Markov chain sampler. Journal of Computational and Graphical Statistics, 26.3: 738-744. https://doi.org/10.1080/10618600.2017.1336446")
                          ),
                          h4("Books"),
                          tags$ul(
                            tags$li("S.P. Meyn and R.L. Tweedie (1993), Markov chains and stochastic stability. Springer-Verlag, London. https://doi.org/10.1017/CBO9780511626630"),
                            tags$li("S. Ross (2014), Introduction to Probability Models (Eleventh Edition). Academic Press. https://doi.org/10.1016/B978-0-12-407948-9.00012-8")
                          ),
                          h4("Websites"),
                          tags$ul(
                            tags$li("Geyer, C. Burn-In is Unnecessary. University of Minnesota Twin Cities, http://users.stat.umn.edu/~geyer/mcmc/burn.html#meyn. Accessed 3 May 2020."),
                            tags$li("Schmidt, D. Stochastic Models and Simulation Course Website. University of Nevada, Reno, https://www.deenaschmidt.com/Teaching/Sp20/Stat753/ Accessed 27 April 2020.")
                          ),
                          h4("R Packages"),
                          tags$ul(
                            tags$li("shiny"),
                            tags$li("shinythemes"),
                            tags$li("shinycustomloader"),
                            tags$li("gganimate"),
                            tags$li("ggplot2"),
                            tags$li("gifski"),
                            tags$li("tidyr"),
                            tags$li("transformr")
                          ),
                          
                 )
###################################################################
                 
                 
,fluid=FALSE                 
)


#MCMC Functions
###########################################################

#Uniform Proposal & Normal Target 
###################################
MH_norm <- function(n=100, alpha=1, x_initial=0, mu=0, sd=1) 
{
  samples <- vector("numeric", n)
  x <- x_initial
  samples[1] <- x
  for (i in 2:n) {
    increment <- runif(1, -alpha, alpha)
    proposal <- x + increment
    aprob <- min(1, dnorm(proposal,mu,sd)/dnorm(x,mu,sd))
    u <- runif(1)
    if (u < aprob){
      x <- proposal
    } 
    samples[i] <- x
  }
  samples
}

#Normal Proposal & Normal Target 
###################################
MH_norm2 <- function(n=100, x_initial=0, mup=0, sdp=1, mut=0, sdt=1)
{
  samples = numeric(n)  
  samples[1] = x_initial      # Initial guess (reasonable)
  
  for (i in 2:n){
    proposal = samples[i-1] + rnorm(1,mup,sdp)  # Proposal value
    if ((dnorm(proposal,mut,sdt) / dnorm(samples[i-1],mut,sdt)) > runif(1))
      samples[i] = proposal                 # Accept proposal value
    else (samples[i] = samples[i-1])        # Reject proposal value
  }
  samples  
}

#Normal Proposal & Gamma Target 
###################################
MH_norm3 <- function(n=100, mup=0, sdp=1, a=2.3, b=2.7) 
{
  mu <- a/b
  sigma <- sqrt(a/(b*b))
  samples <- vector("numeric", n)
  x <- a/b
  samples[1] <- x  # initial value is the mean of the gamma distribution
  for (i in 2:n) {
    proposal <- x + rnorm(1,mup,sdp)  
    aprob <- min(1, (dgamma(proposal,a,b)/dgamma(x,a,b)))
    u <- runif(1)
    if (u < aprob){
      x <- proposal
    } 
    samples[i] <- x
  }
  samples
}


#Server
###################################################################
server <- function(input, output) {
 
#Normal Proposal & Normal Target 
###################################
  
  data1 <- function(){
    run1 <- MH_norm(n=input$iter1,alpha=input$p1alpha,x_initial=input$x1_initial,mu=input$t1mu,sd=input$t1sdev)
    sample_df <- as.data.frame(cbind(1:length(run1),run1,0))
    colnames(sample_df) <- c("x","y","anim_index")
    sample_df
  }
  
  data2 <- function(){
    run1 <- MH_norm2(n=input$iter2, x_initial=input$x2_initial, mup=input$p2mu, sdp=input$p2sdev, mut=input$t2mu, sdt=input$t2sdev)
    sample_df <- as.data.frame(cbind(1:length(run1),run1,0))
    colnames(sample_df) <- c("x","y","anim_index")
    sample_df
  }
  
  data3 <- function(){
    run1 <- MH_norm3(n=input$iter3, mup=input$p3mu, sdp=input$p3sdev, a=input$t3a, b=input$t3b) 
    sample_df <- as.data.frame(cbind(1:length(run1),run1,0))
    colnames(sample_df) <- c("x","y","anim_index")
    sample_df
  }
  
  # data4 <- function(){
  #   run1 <- MH_norm(n=200,alpha=1,x_initial=8,mu=0,sd=0.5)
  #   run2 <- MH_norm(n=200,alpha=1,x_initial=0,mu=0,sd=0.5) 
  #   sample_df <- as.data.frame(cbind(1:length(run1),run1,run2,0))
  #   colnames(sample_df) <- c("x","y1","y2","y3")
  #   sample_df
  # }
  
  hist_reactive1 <- eventReactive(input$go1,
    {
    outfile <- tempfile(fileext='.gif')

    sample_df <- data1()
    
    #Histogram Breakdown of Animation Index
    
    x <- nrow(sample_df)
    part1 <- as.integer(1*(x/100))
    part2 <- part1 + as.integer(3*(x/100))
    part3 <- part2 + as.integer(5*(x/100))
    part4 <- part3 + as.integer(7*(x/100))
    part5 <- part4 + as.integer(9*(x/100))
    part6 <- part5 + as.integer(11*(x/100))
    part7 <- part6 + as.integer(13*(x/100))
    part8 <- part7 + as.integer(15*(x/100))
    part9 <- part8 + as.integer(17*(x/100))
    part10 <- part9 + as.integer(19*(x/100))
    
    sample_df[1:part1,"anim_index"] <- 1
    sample_df[(part1+1):part2,"anim_index"] <- 2
    sample_df[(part2+1):part3,"anim_index"] <- 3
    sample_df[(part3+1):part4,"anim_index"] <- 4
    sample_df[(part4+1):part5,"anim_index"] <- 5
    sample_df[(part5+1):part6,"anim_index"] <- 6
    sample_df[(part6+1):part7,"anim_index"] <- 7
    sample_df[(part7+1):part8,"anim_index"] <- 8
    sample_df[(part8+1):part9,"anim_index"] <- 9
    sample_df[(part9+1):part10,"anim_index"] <- 10
  
    
    # now make the animation
    p = ggplot(sample_df, aes(x=y)) + 
      geom_histogram(bins=30, fill="#69b3a2", color="#e9ecef", alpha=0.9)+
      labs(title="Sample Histogram",x = 'Sample values', y = 'Count') + 
      theme_minimal() + 
      coord_flip() +
      scale_y_reverse() +
      transition_states(anim_index)
    
    anim_save("outfile1.gif", animate(p,renderer = gifski_renderer())) # New
    
    # Return a list containing the filename
    list(src = "outfile1.gif",
         contentType = 'image/gif'
         # width = 400,
         # height = 300,
         # alt = "This is alternate text"
    )}
  )
  
  plot_reactive1 <- eventReactive(input$go1,
    {
    outfile <- tempfile(fileext='.gif')
                                    
    sample_df <- data1()

    # now make the animation
    p = ggplot(sample_df, aes(x=x, y=y)) + 
    geom_line(color="#69b3a2", alpha=0.9) + 
    geom_segment(aes(xend = x, yend = y), linetype = 2) + 
    geom_point(size = 2,color="#69b3a2", alpha=0.9) + 
    transition_reveal(x) + 
    coord_cartesian(clip = 'off') + 
    labs(title="Sample Path",x = 'Iteration', y="") + 
    theme_minimal()
                                    
    anim_save("outfile2.gif", animate(p, renderer = gifski_renderer())) # New
                                    
    # Return a list containing the filename
    list(src = "outfile2.gif",
    contentType = 'image/gif'
    # width = 400,
    # height = 300,
    # alt = "This is alternate text"
    )}
  )  
  
  
  hist_reactive2 <- eventReactive(input$go2,
  {
    outfile <- tempfile(fileext='.gif')
    
    sample_df <- data2()
    
    #Histogram Breakdown of Animation Index
    
    x <- nrow(sample_df)
    part1 <- as.integer(1*(x/100))
    part2 <- part1 + as.integer(3*(x/100))
    part3 <- part2 + as.integer(5*(x/100))
    part4 <- part3 + as.integer(7*(x/100))
    part5 <- part4 + as.integer(9*(x/100))
    part6 <- part5 + as.integer(11*(x/100))
    part7 <- part6 + as.integer(13*(x/100))
    part8 <- part7 + as.integer(15*(x/100))
    part9 <- part8 + as.integer(17*(x/100))
    part10 <- part9 + as.integer(19*(x/100))
    
    sample_df[1:part1,"anim_index"] <- 1
    sample_df[(part1+1):part2,"anim_index"] <- 2
    sample_df[(part2+1):part3,"anim_index"] <- 3
    sample_df[(part3+1):part4,"anim_index"] <- 4
    sample_df[(part4+1):part5,"anim_index"] <- 5
    sample_df[(part5+1):part6,"anim_index"] <- 6
    sample_df[(part6+1):part7,"anim_index"] <- 7
    sample_df[(part7+1):part8,"anim_index"] <- 8
    sample_df[(part8+1):part9,"anim_index"] <- 9
    sample_df[(part9+1):part10,"anim_index"] <- 10
    
    
    # now make the animation
    p = ggplot(sample_df, aes(x=y)) + 
      geom_histogram(bins=30, fill="steelblue3", color="#e9ecef", alpha=0.9)+
      labs(title="Sample Histogram",x = 'Sample values', y = 'Count') + 
      theme_minimal() + 
      coord_flip() +
      scale_y_reverse() +
      transition_states(anim_index)
    
    anim_save("outfile3.gif", animate(p, renderer = gifski_renderer())) # New
    
    # Return a list containing the filename
    list(src = "outfile3.gif",
         contentType = 'image/gif'
         # width = 400,
         # height = 300,
         # alt = "This is alternate text"
    )}
  )
  
  plot_reactive2 <- eventReactive(input$go2,
    {
      outfile <- tempfile(fileext='.gif')
      
      sample_df <- data2()
      
      # now make the animation
      p = ggplot(sample_df, aes(x=x, y=y)) + 
        geom_line(color="steelblue3", alpha=0.9) + 
        geom_segment(aes(xend = x, yend = y), linetype = 2) + 
        geom_point(size = 2,color="steelblue3", alpha=0.9) + 
        transition_reveal(x) + 
        coord_cartesian(clip = 'off') + 
        labs(title="Sample Path",x = 'Iteration', y="") + 
        theme_minimal()
      
      anim_save("outfile4.gif", animate(p, renderer = gifski_renderer())) # New
      
      # Return a list containing the filename
      list(src = "outfile4.gif",
           contentType = 'image/gif'
           # width = 400,
           # height = 300,
           # alt = "This is alternate text"
      )}
  )
  
  
  hist_reactive3 <- eventReactive(input$go3,
    {
      outfile <- tempfile(fileext='.gif')
      
      sample_df <- data3()
      
      #Histogram Breakdown of Animation Index
      
      x <- nrow(sample_df)
      part1 <- as.integer(1*(x/100))
      part2 <- part1 + as.integer(3*(x/100))
      part3 <- part2 + as.integer(5*(x/100))
      part4 <- part3 + as.integer(7*(x/100))
      part5 <- part4 + as.integer(9*(x/100))
      part6 <- part5 + as.integer(11*(x/100))
      part7 <- part6 + as.integer(13*(x/100))
      part8 <- part7 + as.integer(15*(x/100))
      part9 <- part8 + as.integer(17*(x/100))
      part10 <- part9 + as.integer(19*(x/100))
      
      sample_df[1:part1,"anim_index"] <- 1
      sample_df[(part1+1):part2,"anim_index"] <- 2
      sample_df[(part2+1):part3,"anim_index"] <- 3
      sample_df[(part3+1):part4,"anim_index"] <- 4
      sample_df[(part4+1):part5,"anim_index"] <- 5
      sample_df[(part5+1):part6,"anim_index"] <- 6
      sample_df[(part6+1):part7,"anim_index"] <- 7
      sample_df[(part7+1):part8,"anim_index"] <- 8
      sample_df[(part8+1):part9,"anim_index"] <- 9
      sample_df[(part9+1):part10,"anim_index"] <- 10
      
      
      # now make the animation
      p = ggplot(sample_df, aes(x=y)) + 
        geom_histogram(bins=30, fill="orangered3", color="#e9ecef", alpha=0.9)+
        labs(title="Sample Histogram",x = 'Sample values', y = 'Count') + 
        theme_minimal() + 
        coord_flip() +
        scale_y_reverse() +
        transition_states(anim_index)
      
      anim_save("outfile5.gif", animate(p, renderer = gifski_renderer())) # New
      
      # Return a list containing the filename
      list(src = "outfile5.gif",
           contentType = 'image/gif'
           # width = 400,
           # height = 300,
           # alt = "This is alternate text"
      )}
  )
  
  plot_reactive3 <- eventReactive(input$go3,
      {
        outfile <- tempfile(fileext='.gif')
        
        sample_df <- data3()
        
        # now make the animation
        p = ggplot(sample_df, aes(x=x, y=y)) + 
          geom_line(color="orangered3", alpha=0.9) + 
          geom_segment(aes(xend = x, yend = y), linetype = 2) + 
          geom_point(size = 2,color="orangered3", alpha=0.9) + 
          transition_reveal(x) + 
          coord_cartesian(clip = 'off') + 
          labs(title="Sample Path",x = 'Iteration', y="") + 
          theme_minimal()
        
        anim_save("outfile6.gif", animate(p, renderer = gifski_renderer())) # New
        
        # Return a list containing the filename
        list(src = "outfile6.gif",
             contentType = 'image/gif'
             # width = 400,
             # height = 300,
             # alt = "This is alternate text"
        )}
  )
  
  plot_unreactive1 <-
    {
        #outfile <- tempfile(fileext='.gif')
        
        # sample_df <- data4()
        # 
        # p=ggplot(sample_df, aes(x=x, y=y1)) + 
        #   geom_line(alpha=0.5,color="dodgerblue4") + 
        #   geom_segment(aes(xend = x, yend = y1), linetype = 2) + 
        #   transition_reveal(x) + 
        #   coord_cartesian(clip = 'off') + 
        #   labs(title="Sub-Optimal Initial Value",x = 'Iteration', y = 'Sampled Values') + 
        #   geom_point(color="dodgerblue4") +
        #   theme_minimal()+
        #   lims(y = c(-8, 8))
        
        # anim_save("outfile7.gif", animate(p, renderer = gifski_renderer())) # New
        
        # Return a list containing the filename
        list(src = "outfile7.gif",
             contentType = 'image/gif'
              #width = 500,
              #height = 450
             # alt = "This is alternate text"
        )}
  
  plot_unreactive2 <-
    {
      # outfile <- tempfile(fileext='.gif')
      
      # sample_df <- data4()
      # 
      # p=ggplot(sample_df, aes(x=x, y=y2)) + 
      #   geom_line(alpha=0.5,color="red4") + 
      #   geom_segment(aes(xend = x, yend = y2), linetype = 2) + 
      #   transition_reveal(x) + 
      #   coord_cartesian(clip = 'off') + 
      #   labs(title="Optimal Initial Value",x = 'Iteration', y = '') + 
      #   geom_point(color="red4") +
      #   theme_minimal()+
      #   lims(y = c(-8, 8))
      
      # anim_save("outfile8.gif", animate(p, renderer = gifski_renderer())) # New
      
      # Return a list containing the filename
      list(src = "outfile8.gif",
           contentType = 'image/gif'
           #width = 500,
           #height = 450
           # alt = "This is alternate text"
      )}
  
  plot_unreactive3 <-
    {
      # outfile <- tempfile(fileext='.gif')
      # 
      # sample_df <- data4()
      # 
      # p=ggplot(sample_df, aes(x=x, y=y3)) + 
      #   geom_line(alpha=0.9,color="springgreen4") + 
      #   geom_segment(aes(xend = x, yend = y3)) + 
      #   transition_reveal(x) + 
      #   coord_cartesian(clip = 'off') + 
      #   labs(title="Target Distribution Mean",x = 'Iteration', y = '') + 
      #   geom_point(color="springgreen4") +
      #   theme_minimal()+
      #   lims(y = c(-8, 8))
      
      # anim_save("outfile9.gif", animate(p, renderer = gifski_renderer())) # New
      
      # Return a list containing the filename
      list(src = "outfile9.gif",
           contentType = 'image/gif'
           #width = 500,
           #height = 450
           # alt = "This is alternate text"
      )}
  
  plot_reactive4 <- eventReactive(input$go5,
    {
      outfile <- tempfile(fileext='.gif')
      
      m <- 0
      s <- 1
      
      samples <- rnorm(1000, m, s)
      
      cummean <- function(x)
        cumsum(x) / seq_along(x)
      
      df <- data.frame()[1:1000, ]
      
      for(i in 1:30){
        df[,i] <- cummean(rnorm(1000, 0, 1))
      }
      
      df$iter <- 1:1000
      
      df_long <- gather(df, run, value,V1:V30, factor_key=TRUE)
      
      head(df_long)
      
      i<-input$iterb1
      
      df_long_mod <- df_long[df_long$iter>i,]
      
      p=ggplot(df_long_mod, aes(iter, value, color = run),show.legend = FALSE) + 
        geom_line(show.legend = FALSE) + 
        geom_point(x = 1000.1,show.legend = FALSE) + 
        transition_reveal(iter) + 
        coord_cartesian(clip = 'off') + 
        labs(title = '', y = 'Sample Values',x='Number of iterations') +
        ylim(-2.5,2.5) +
        theme_minimal()
      
      anim_save("outfile10.gif", animate(p, renderer = gifski_renderer())) # New
      
      # Return a list containing the filename
      list(src = "outfile10.gif",
           contentType = 'image/gif'
           #width = 500,
           #height = 450
           # alt = "This is alternate text"
      )}
  )
  
  output$hist1 <- renderImage(hist_reactive1(),deleteFile = TRUE)
  output$plot1 <- renderImage(plot_reactive1(),deleteFile = TRUE)
  
  output$hist2 <- renderImage(hist_reactive2(),deleteFile = TRUE)
  output$plot2 <- renderImage(plot_reactive2(),deleteFile = TRUE)
  
  output$hist3 <- renderImage(hist_reactive3(),deleteFile = TRUE)
  output$plot3 <- renderImage(plot_reactive3(),deleteFile = TRUE)
  
  output$plot4 <- renderImage(plot_unreactive1,deleteFile = FALSE)
  output$plot5 <- renderImage(plot_unreactive2,deleteFile = FALSE)
  output$plot6 <- renderImage(plot_unreactive3,deleteFile = FALSE)
  
  output$plot7 <- renderImage(plot_reactive4(),deleteFile = TRUE)
  
  output$static1 <- renderPlot({
    set.seed(11)
    run1 <- MH_norm()
    autocorr <- acf(run1, plot = FALSE)
    autocorr_df <- with(autocorr, data.frame(lag, acf))
    
    ggplot(data = autocorr_df, mapping = aes(x = lag, y = acf)) +
      geom_hline(aes(yintercept = 0)) +
      geom_segment(mapping = aes(xend = lag, yend = 0)) +
      theme_minimal() +
      xlim(0, 20)+
      ylim(-0.25,1)+
      labs(title = 'Strong Autocorrelation for increasing lags') 
  })
  
  output$static2 <- renderPlot({
    set.seed(123)
    x <- arima.sim(n = 200, model = list(ar = 0.6))
    bacf <- acf(x, plot = FALSE)
    bacfdf <- with(bacf, data.frame(lag, acf))
    
    ggplot(data = bacfdf, mapping = aes(x = lag, y = acf)) +
      geom_hline(aes(yintercept = 0)) +
      geom_segment(mapping = aes(xend = lag, yend = 0))+
      theme_minimal()+
      xlim(0, 20)+
      ylim(-0.25,1)+
      labs(title = 'Weak Autocorrelation for increasing lags') 
  })
  
  
  output$interactive1 <- renderPlot({
    set.seed(12)
    run1 <- MH_norm(n=10000)
    run1_mod <- run1[seq(1, length(run1),input$itert1)]
    autocorr <- acf(run1_mod, plot = FALSE)
    autocorr_df <- with(autocorr, data.frame(lag, acf))
    
    ggplot(data = autocorr_df, mapping = aes(x = lag, y = acf)) +
      geom_hline(aes(yintercept = 0)) +
      geom_segment(mapping = aes(xend = lag, yend = 0)) +
      theme_minimal()
    
  })
  
}

#App Construction
###################################################################
shinyApp(server = server, ui = ui)