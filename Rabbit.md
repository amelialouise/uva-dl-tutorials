# R-rabbit holes

Using the RStudio IDE to run Python code for this project has me
scurrying down some rabbit holes anytime something doesn’t work or
“works” in an unexpected way. So this notebook will document references
and thingamabobs that helped me to escape them for a bit.

# Rendering figures in RStudio

In Tutorial 3 we started to render some figures that contained multiple
graphs, e.g.

[![Wowzers.](images/tut3-visual.PNG)](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial3/Activation_Functions.html#Visualizing-the-gradient-flow-after-initialization)

These often did not render well in an R notebook chunk by default.
Here’s that same set of graphs below, for instance.

<figure>
<img src="images/tut3-visual-rchunk-bad.PNG" alt="No bueno" />
<figcaption aria-hidden="true">No bueno</figcaption>
</figure>

I learned in a `{reticulate}` [issue
thread](https://github.com/rstudio/reticulate/issues/1140#issuecomment-1625607199)
that you can modify the figure size in a `Python` chunk by using
`fig.width` and `fig.height` inline, as shown below. I thought that was
kinda neat.

![That's better.](images/tut3-visual-rchunk-better.PNG)It was possible
for me to then move on with ~~my life~~ the tutorials after fiddling
with these options to get a decent-looking-enough figure rendered in the
R notebook. Huzzah!

That was until I knitted the notebook and discovered that the fiddling
did not carry over. 😖 This issue is well-documented in this [closed and
triaged
issue](https://github.com/rstudio/rstudio/issues/4521#issuecomment-1414371481).
There seems to be ongoing work to fix numerous RStudio rendering issues,
[see here](https://github.com/rstudio/rstudio/issues/12740).

So perhaps this will be fixed in the future releases of RStudio? Maybe
you won’t (or didn’t) encounter it yourself because you’re using some
(magical) combination of an RStudio build + OS + knitr/R version. Or
you’re just magical.

Here’s my combo for reference.

[![This is an image because knitr can't render if you use
RStudio.Version()](images/version-info.PNG)](https://community.rstudio.com/t/rstudio-version-not-found-on-knit/8088/3)
