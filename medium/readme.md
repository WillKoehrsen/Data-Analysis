# Tools for analyzing Medium article statistics

* Basic usage:

1. Go to the stats page https://medium.com/me/stats
2. Scroll all the way down to the bottom so all the articles are loaded
3. Right click, and hit 'save as'
4. Save the file as `stats.html` in the `data/` directory. You can also save the responses to do a similar analysis.

![](images/stats-saving-medium.gif)

* Open up a Jupyter Notebook or Python terminal in the `medium/` directory
and run

```
from retrieval import get_data
df = get_data(fname='stats.html')
```

* Note: running on Mac may first require setting
    `export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES`
    from the command line [to enable parallel processing](https://stackoverflow.com/questions/50168647/multiprocessing-causes-python-to-crash-and-gives-an-error-may-have-been-in-progr)
* For complete usage refer to `Medium Stats Analysis`
* Data retrieval code lives in `retrieval.py`
* Visualization and analysis code is in `visuals.py`
* See also the Medium article ["Medium Analysis in Python"]()
* Contributions are welcome and appreciated
* For help contact wjk68@case.edu or twitter.com/@koehrsen_will