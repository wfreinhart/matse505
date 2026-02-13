# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# ---
# id: Lecture01_a_note_about_the_interpreter
# type: Foundational
# parent_lecture: Lecture01
# ---
#
# ## A note about the interpreter
#
# There's something important to know about the messages Python gives us after we write commands.
# In the cells up above, we were using `print` to get the values of our variables.
# Then when we were looking at `type`, we just wrote `type(x)` and got some output.
# But we will only get to see the **last** thing that happened if we don't use `print`.
# Let's take a look:
#
# We called the `type` function twice, but we only got one output.
# Notice that the `type` that got written out is the latest one, from `y='a'`.
# Now let's use `print()` to record each thing that happened:
#
# Not only do we get the output from both `type` calls, but `print` also automatically adds some additional information to the output.
# Here it tells us that the variable belongs to certain builtin `classes`.
# More on that in a minute...
#
#

# %%
y = 1.0  # set y to a floating point number
type(y)  # write out the type of y
y = 'a'  # reset y to a string
type(y)  # write out the type of y

y = 1.0         # set y to a floating point number
print(type(y))  # explicitly print out the type of y
y = 'a'         # reset y to a string
print(type(y))  # explicitly print out the type of y
