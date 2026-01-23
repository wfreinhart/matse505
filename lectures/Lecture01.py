# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -id,-colab,-outputId
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.0
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # Introduction

# %% [markdown]
# <img src="../lectures/assets/lecture01_xkcd_python.jpg" alt="XKCD comic #353 about Python: a character learns Python in minutes and realizes they can fly." height=500/>
#
# (image credit [the XKCD website](https://xkcd.com/353/))
#

# %% [markdown]
# ## Python language
#
# In this class we will use the [Python programming language](https://en.wikipedia.org/wiki/Python_(programming_language)).
# Python is free and open source, and is distibuted under a [license](https://www.python.org/about/) which allows its users to use and distribute it freely, even for commercial use.
# This means you can bring it with you to work no matter where you end up!
#
# *What is Python?*
#
# Python is a general purpose programming language that has found uses in all sorts of contexts from education to manufacturing.
# It is a powerful language which powers real data science workflows for Fortune 500 companies.
# At the same time, Python was designed to be more human-readable than other languages like C++ and Java.
# It uses different syntax than those languages to give programs a clear, logical flow and a preference for words over symbols.
# Python is also [object oriented](https://en.wikipedia.org/wiki/Object-oriented_programming), but we won't talk about that in detail until much later in the course.
#
# *How do you get Python?*
#
# There are a lot of different distributions of Python available, including a very popular one called [Anaconda](https://www.anaconda.com/products/individual) that you will probably run across in internet searches during this class.
# Typically to use Python you need to install it on your local machine.
# The Anaconda installer is about 500 MB, so this is not a trivial ask (although Mac OS comes with a Python distribution pre-installed).

# %% [markdown]
# ## Google Colaboratory
#
# <img src="../lectures/assets/lecture01_colab_logo.jpg" width=200 alt="The Google Colab logo, consisting of a stylized infinity loop in Google colors.">
#
# *What are we doing on this site?*
#
# It turns out there are other ways to use Python besides downloading and installing it.
# Google has provided a basic Python environment for everyone to use free of charge through the Colaboratory platform, or Colab for short.
# This web page is hosted on [the Google Colab website](https://colab.research.google.com).
# You can create your own such pages for free by going to File ➡ New notebook.
#
# **Important:** you must be logged in with a Google account to run code in Colab.
# The best thing to do is log in with your Penn State credentials so we know whose work it is when you submit your Notebooks.
#
# **Everyone should log in now since we'll need to run code shortly!**

# %% [markdown]
# ## Jupyter notebooks
#
# *What's a Notebook?*
#
# We're inside a Jupyter notebook right now!
# A Jupyter Notebook is a more flexible and user-friendly way to write and read Python code.
# [Jupyter](https://en.wikipedia.org/wiki/Project_Jupyter) provides a sort of software wrapper around the Python ["kernel"](https://en.wikipedia.org/wiki/Kernel_(operating_system)) and acts as a go-between to provide some quality of life enhancements for the user (you!).

# %% [markdown]
# Jupyter lets us include **formatted text** and $E_{Q}u^at_io^ns$ alongside [hyperlinks](https://en.wikipedia.org/wiki/Project_Jupyter) and images:
#
# <img src="../lectures/assets/python_logo.png" alt="The official Python logo, featuring two interlocking snakes in blue and yellow." height=200/>
#
# <img src="../lectures/assets/jupyter_logo.png" alt="The Project Jupyter logo, showing an orange planet-like circle with orbiting moons." height=200/>
#
#

# %%
# jupyter also lets us embed python code right alongside all that
# this is a live python code cell!

# %% [markdown]
# The notebook format lets us seamlessly integrate code and non-code blocks so we can easily create tutorials.
# This has actually become an industry standard for data science researchers, and software distributions are basically expected to include `.ipynb` walkthroughs in addition to the source code.
#
# A great example is Google's very own [Colab tutorial](https://colab.research.google.com/notebooks/intro.ipynb).
# Let's take a look...
#
# ...you may see there are plenty of things we won't cover such as the machine learning package Tensorflow and accelerated hardware like Graphics Processing Units (GPUs) and Tensor Processing Units (TPUs).
# However, we will cover basic Python syntax, working with data, and making charts as shown.

# %% [markdown]
# When you look at plain Python source code, you will see a distinct difference.
# Source code does usually include comments like the green text above (after the `#` sign), but not any other elements like rich text, images, or actual code output.
# The picture below illustrates the difference:

# %% [markdown]
# ![A comparison diagram showing a Jupyter Notebook with mixed text and code vs a plain Python script with only code.](../lectures/assets/lecture01_notebooks_vs_scripts.jpg)
#
# (image credit [Krishna Subramanian](https://krishna-subramanian.net/index.php/portfolio/notebooksvscripts/))

# %% [markdown]
# # Python syntax

# %% [markdown]
# With the basics out of the way, let's talk about how to actually write and run Python code!
#
# [Python syntax](https://www.w3schools.com/python/python_syntax.asp) will follow a recipe like `variable_name = value` to create or modify variables, `function(variable)` to run functions or methods, or `#` to leave a comment.
# Here's an example with all three elements:

# %%
# let's print a number:
x = 5
print(x)

# %% [markdown]
# **Note:** The grey box indicates a *Code Cell* as opposed to the *Text Cell* that we've been using so far.
# It's actual Python code and can be run directly in your browser (thanks to Google Colab) to see the result.
#
# Go ahead and run the code above by clicking the arrow that appears inside the brackets `[ ]` on the left side of the cell or by pressing `Shift + Enter` with the cell selected.
# It should just give the same answer again because we stored the output.

# %% [markdown]
# ## Comments
#
# The `#` symbol creates a [comment](https://www.w3schools.com/python/python_comments.asp) in the code.
# A comment is text that does not get interpreted by the language.
# Instead, it improves readability by providing an explanation or reference for a particular line of code.
#
# Comments are incredibly helpful and it is actually standard practice to include them to guide other people in using your code.
# Because everyone thinks and codes a little differently, it can really help communicate ideas.
# Here's another example:

# %%
# comments can be on just one line
"""
comments can also be on multiple lines
just use the triple " sign to switch to comment mode
(and again to switch back)
"""
x = 5
print(x)

# %% [markdown]
# ## Getting help

# %% [markdown]
# You can get information about functions using `help`:

# %%
help(print)

# %% [markdown]
# Another option is to use the `?` symbol, which will open a tab in the Colab interpreter (look for a `Help` tab to the right ➡)

# %%
# print?

# %% [markdown]
# Note that this output might not be helpful to *you* until you learn more about how Python works...
# You can always refer to online documentation for a more human-readable version (often including examples, etc...)

# %% [markdown]
# ## Tab completion
#
# A handy feature of the Colab interpreter is the use of auto-complete.
# You can use `<Tab>` to complete a partial statement based on functions or variables the interpreter is already aware of.
# Try typing `pri` and wait a second for the auto-complete menu to pop up.
# Then press `<Tab>` to complete the statement to `print`:

# %%

# %% [markdown]
# ## Variables

# %% [markdown]
# ### Creating variables
#
# Anything we assign a value to becomes a [variable](https://www.w3schools.com/python/python_variables.asp).
# Variables are created when you first assign something to them and persist until the program completes or you manually delete them.
# From that point forward you can reference the variable to access the stored value.
#
# > There is something called scope that will make this statement more nuanced. We'll talk about it later.
#
# Let's try creating a variable in a few different ways:
#

# %%
x = 1  # this changes the value from before!
my_variable = "abc"

# %% [markdown]
# Variables can have any name that follows these rules:
# * not a [reserved keyword](https://www.w3schools.com/python/python_ref_keywords.asp) (one of the builtin parts of the language)
# * no spaces
# * starts with a letter (not a number)
#
# Variables are also *case sensitive*.
# For variables that make sense as multiple words, we usually use an underscore `_` instead of a space.
# Here are some examples of valid variable names:

# %%
a_variable_with_a_long_name = 1
MyMixedCaseVariable = 2
c = 3
my_var_4 = 4

print(a_variable_with_a_long_name)
print(MyMixedCaseVariable)
print(c)
print(my_var_4)

# %% [markdown]
# Let's see what happens if we don't respect the case sensitivity:

# %%
print(mymixedcasevariable)

# %% [markdown]
# Oops, we get a `NameError`.
# This is Python trying to tell us that it doesn't understand our command in as much detail as it can.
# We'll learn more about [Errors and Exceptions](https://docs.python.org/3/tutorial/errors.html) later.

# %% [markdown]
# ### Manipulating variables
#
# We can manipulate variables using math.
# You can imagine Python as a calculator if you're just using some of the basic functionality:

# %%
x = 1 * 2  # x will be 2
x = x + 3  # x will be 2 + 3 = 5
print(x)   # print the value to see

# %% [markdown]
# Multiple variables can be involved in these math operations:

# %%
x = 1
y = 2
z = 3
print(x * y + z)

# %% [markdown]
# Once a variable is defined, its value persists between code blocks.
# For example, do remember what value we assigned to `a_variable_with_a_long_name` from before?
# The Python interpreter does!

# %%
print(a_variable_with_a_long_name)
print(a_variable_with_a_long_name * 2)
print(a_variable_with_a_long_name + 5)

# %% [markdown]
# When manipulating a variable like this, it can get tiring to write expressions like `my_variable = my_variable + 1`.
# We can use a shorthand `+=` or `-=` to *increment* or *decrement* the value like so:

# %%
x = 1
print(x)
x += 1
print(x)
x += x  # we can also use variable values
print(x)
x *= 2  # we can use +, -, *, and / this way
print(x)

# %% [markdown]
# # Functions
#
# [Functions](https://www.w3schools.com/python/python_functions.asp) do something to variables.
# You may have learned in math about functions like $f(x) = y$.
# The Python version of a function is exactly the same, it produces some output $y$ from some input $x$.
#
# There are a bunch of built-in functions including `print` that we have already seen.
# Let's try using print in a few different ways:

# %%
print(x)            # this will write "1" (from x = 1 above)
print(my_variable)  # this will write "abc" (from my_variable = 'abc' above)
print(5)
print("some text")  # we can also call print on an input that's not a variable
print(x, 'text')    # print can take multiple arguments separated by commas

# %%
print(print)

# %% [markdown]
# Functions are called by writing the name followed by `()`.
# The function `arguments` go inside the parentheses.
# Some functions take no arguments and would look like this: `my_function()`.
#
# Another builtin function is `type`:

# %%
type(x)  # x should be of type "integer," or "int" for short

# %% [markdown]
# Note that `x` is an integer because its assigned value did not include a decimal point.
# If we use a decimal point, it will become a floating point number (`float`):

# %%
x = 1.0
type(x)

# %% [markdown]
# ## A note about the interpreter
#
# There's something important to know about the messages Python gives us after we write commands.
# In the cells up above, we were using `print` to get the values of our variables.
# Then when we were looking at `type`, we just wrote `type(x)` and got some output.
# But we will only get to see the **last** thing that happened if we don't use `print`.
# Let's take a look:

# %%
y = 1.0  # set y to a floating point number
type(y)  # write out the type of y
y = 'a'  # reset y to a string
type(y)  # write out the type of y

# %% [markdown]
# We called the `type` function twice, but we only got one output.
# Notice that the `type` that got written out is the latest one, from `y='a'`.
# Now let's use `print()` to record each thing that happened:

# %%
y = 1.0         # set y to a floating point number
print(type(y))  # explicitly print out the type of y
y = 'a'         # reset y to a string
print(type(y))  # explicitly print out the type of y


# %% [markdown]
# Not only do we get the output from both `type` calls, but `print` also automatically adds some additional information to the output.
# Here it tells us that the variable belongs to certain builtin `classes`.
# More on that in a minute...

# %% [markdown]
#

# %% [markdown]
# ## Defining functions
#
# Using functions from open source modules will save us a lot of time and effort by providing well known formulas and functionalities without us having to design them ourselves.
# Here's how we can define our own functions with `def`:

# %%
# def is the python keyword
# my_function is the name
# arg is a positional argument
def my_function(arg):
    # do something with the argument
    print(arg)
    # the extent is indicated by indent

# call the function outside the indent
out = my_function('hello')
print(out)

# %% [markdown]
# This function does not send the value back to us, it only prints something. We can make this behavior more explicit like so:

# %%
out = print('hello')
print(out)

# %%
out = my_function('hello')
print(f'out = {out}')


# %% [markdown]
# To keep the value, we need to use `return`:

# %%
def my_function(arg):
    arg_plus = arg + ' from my_function'
    print(arg_plus)
    return arg_plus  # this gets sent back to the main program

out = my_function('hello')  # out will be set to `arg_plus`
print(out)


# %% [markdown]
# ## Arguments
#
# We have already discussed the difference between *positional* and *keyword* arguments when using existing functions. Now we will see how it works when we define our own.

# %% [markdown]
# ### Positional arguments
#
# Here we define a function with two positional arguments. Which input value goes to `a` and which goes to `b` is therefore determined by the *order* of the inputs:

# %%
def my_function(a, b):
    return (a + 1) * b

print(my_function(1, 2))  # a = 1, b = 2
print(my_function(2, 1))  # a = 2, b = 1


# %% [markdown]
# There is no limit to the number of positional arguments we provide. However, from a practical perspective it gets annoying to provide more than 3 or 4:

# %%
def my_function(a, b, c, d, e, f, g, h, i, j):
    # this should just take a list!!!
    return a + b + c + d + e + f + g + h + i + j

print(my_function(0, 1, 2, 3, 4, 5, 6, 7, 8, 9))


# %% [markdown]
# ### Returning multiple values
#
# We can also `return` multiple values. We've used functions with this behavior before, like `subplots()` and `pearsonr`. All we need to do is list multiple values separated by commas:

# %%
def my_function(a, b):
    return a + b, a * b

out = my_function(2, 3)
print(out)

plus, mult = out
print(plus)
print(mult)


# %% [markdown]
# This works exactly like positional arguments where the order determines which variable gets which value. Also like positional arguments, we should practically limit the number of values being returned so as not to annoy the user (likely ourselves!):

# %%
def my_function(a):
    return a + 1, a + 2, a + 3, a + 4, a + 5, a + 6

a1, a2, a3, a4, a5, a6 = my_function(1)
print(a1, a2, a3, a4, a5, a6)


# %% [markdown]
# We can also use conditional statements to control which variables get returned.
#
# > Note: this can make the interface to your function rather confusing or inconsistent. Think carefully about this design choice.

# %%
def my_function(a):
    return 0
    # these don't run after "return"
    a += 1
    return a

my_function(10)


# %%
def my_function(a, return_a):
    b = a + 1
    if return_a:
        return a, b
    else:
        return b

print(my_function(1, False))
print(my_function(1, True))


# %% [markdown]
# ### Keyword arguments
#
# In addition to positional arguments, we can use *keywords* to assign specific values to specific variables inside our function. This allows us to:
# * mix the order of inputs
# * give default values so not every parameter needs to be specified
#
# Let's look at an example:

# %%
def my_function(a, b=2):
    return a * b

print(my_function(3))
print(my_function(3, 3))
print(my_function(3, b=4))


# %% [markdown]
# Notice how the value `b = 2` was not provided in the function call, it happens automatically in the function itself. We could also provide a different value and it will use that. Either position or the keyword itself can set the value of `b`.

# %% [markdown]
# We can do the same thing with several keyword arguments:

# %%
def my_function(a, b=2, c=3):
    return a * b + c

print(my_function(2))            # use default values
print(my_function(2, 3, 2))      # in order as positional args
print(my_function(2, c=3, b=2))  # out of order as keyword args!

# %% [markdown]
# We also don't have to specify both values, we can do any combination of them:

# %%
print(my_function(2, c=2))
print(my_function(2, b=2))

# %% [markdown]
# Finally, we can also refer to the positional arguments by their names if we want to reference them out of order:

# %%
# the function def from above:
# def my_function(a, b=2, c=3):
print(my_function(b=2, c=3, a=2))  # refer to a using `a=` keyword!


# %% [markdown]
# ### Sidenote: unpacking arguments
#
# Python also supports a pretty crazy syntax for "argument unpacking." Let's say we put a bunch of arguments in a `list`. Then we could do a function call using each element of this `list` in this way:

# %%
def my_function(a, b, c):
    return a + b + c

args = [1, 2, 3]
my_function(args[0], args[1], args[2])

# %% [markdown]
# What if we wanted to tell `my_function` that the `list` is *the list of arguments*? So instead of `args[0], args[1], args[2]` we could just say "the positional arguments are in order in this `list`. We can do exactly that with the `*` operator:

# %%
my_function(*args)

# %% [markdown]
# # Modules
#
# *What are modules?*
#
# We've gone over some basic Python functionality, but so far it doesn't seem that powerful.
# We can:
# * define variables
# * perform basic math on them (multiply, divide, add, subtract)
# * convert them to different types
# * print their values
#
# This is less than your graphing calculator can do, so what's the big deal?
# Basically, the answer is **modules**.
# Python modules are pieces of code that other people have written and posted on the internet for free distribution.
# By accessing those modules, you can expand Python's functionality to do nearly anything you can think of.
#
# *What are some commonly used modules?*
#
# **NumPy**
#
# [`numpy`](https://numpy.org/) is so widespread it might as well be considered part of the Python language.
# Basically any program with mathematical equations or data manipulation uses NumPy.
# This includes the other modules we're about to look at!
#
# **Matplotlib**
#
# [`matplotlib`](https://matplotlib.org/) is one of the most popular visualization libraries for Python (although there are others).
# Its name comes from its inspiration from the MATLAB plotting library.
# If you've used MATLAB, this will look familiar.
#
# **SciPy**
#
# [`scipy`](https://docs.scipy.org/doc/scipy/reference/) is a more specialized package for scientific computing.
# It includes things like integration, optimization, and statistical distributions.
#
# **Pandas**
#
# [`pandas`](https://pandas.pydata.org/docs/) is a library for reading, writing, and analyzing data.
# It contains wrappers to do a lot of the things these other libraries do (by calling their functions behind the scenes).
# While we will use all four of these in class, you may end up mostly relying on Pandas "in the wild."

# %% [markdown]
# ## Importing modules
#
#
# You can access modules using the `import` keyword followed by the module name.
# Colab has a bunch of modules available by default.
# Let's try importing NumPy:

# %%
import numpy
print(numpy)

# %% [markdown]
# **Note** modules are supposed to have short, lower-case names.
# Numpy was imported using `numpy` rather than NumPy.
#
# The `print` function gives us some information about the module that indicates it loaded successfully.
# Note that just calling `import` doesn't give any output unless there's an error.
# Let's observe this behavior using Matplotlib:

# %%
import matplotlib

# %% [markdown]
# ## Aliases
#
# Even though modules have relatively short names, they sometimes need to be referenced a ton of times (hundreds or thousands).
# In this case, it can be handy to change the name to something even shorter.
# When using `import`, you can rename things with the [`as` keyword](https://www.w3schools.com/python/ref_keyword_as.asp):

# %%
import numpy as np
print(np)

# %% [markdown]
# Now the very short `np` will serve as an alias for the NumPy module.
# You can think of this just like a variable name, where `np` points to the full NumPy module behind the scenes.

# %% [markdown]
# # Everything is an object
#
# In Python, every variable is an **object**.
# This is not the case in all languages, but only in "object-oriented" languages.
#
# What does it mean?
# It means that variables in Python aren't just numbers, a lot of additional information is attached to them.

# %% [markdown]
# ## A builtin example
#
# Consider the `list` data type:

# %%
my_list = [5, 4, 3, 2, 1]
print(my_list)

# %% [markdown]
# At first glance, we may think of `list` as this series of numbers. But we can ask `list` to do things:

# %%
my_list.append(6)
print(my_list)

# %% [markdown]
# Notice how `append` isn't a Python builtin function, but instead it actually comes after `my_list` plus a `.` (dot)
#
# What does this mean?
# It means that `list` is more than just a series of numbers, it is an **object**. That object is capable of *doing things* and it also *knows information about itself*.
#
# Here's an example: `sort`

# %%
my_list.sort()

# %% [markdown]
# Weird, it looks like nothing happened...let's check on `my_list` again:

# %%
print(my_list)

# %% [markdown]
# The `list` is sorted now! What happened?
#
# Well, we asked `my_list` to sort its entries using `sort()`.
# In fact, `sort` is something that all `list` objects know how to do to their own data. Let's see what else `my_list` can do using `dir`:

# %%
dir(my_list)

# %% [markdown]
# Woah, `my_list` can do a lot of things!
#
# What's up with all those `__` entries? Those are `builtins`, and they are named that way deliberately to "hide" them from you as a user. Notice that one of them is called `__len__`, which is suspiciously similar to the `len()` builtin function.
# It turns out that the builtin `len()` simply calls `list.__len__()`, and we can demonstrate that here:

# %%
my_list.__len__()

# %%
len(my_list)

# %% [markdown]
# ## Methods and attributes
#
# Let's build our vocabulary up a bit more.
# At a basic level, objects can have two types of "things" -- methods and attributes.
#
# * **methods** are functions that belong to the object and can access variables inside it
# * **attributes** are variables that belong to the object
#
# Using our `list` example, `append` and `sort` are methods. `list` does not have any attributes. `np.array` does, however:

# %%
import numpy as np  # this is a module, with an alias

my_arr = np.array(my_list)
print(my_arr)

print(my_arr.mean)   # this is a method
print(my_arr.shape)  # this is an attribute

# %% [markdown]
# We can access methods and attributes inside an object using the same `.` notation. The only difference is what is returned: a callable in the case of methods and an object in the case of attributes. If we want to use the `mean` method, for instance, we need to add the `()` as we do for any function.

# %%
print(my_arr.mean())
print(np.mean(my_arr))
