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
# id: Lecture18_interpolation
# type: Foundational
# parent_lecture: Lecture18
# ---
#
# # Interpolation
#
# What's wrong with this? Let's pretend we're especially interested in the region around 25% strain.
#
# If we wanted to read a value off the chart in between 0.25 and 0.26, for instance, we would need to *interpolate* between the known values. Let's remind ourselves what values we know from the chart:
#
# Ok, let's say we want to get the stress at 0.253 % strain. How can we do it? We need to make in informed guess at what it would be based on the values around it. The simplest strategy could be to assume it is the same value as the closest known result. In this case, it would be:
#
# So we're saying that because we know the stress is 11.75 MPa at 0.250 strain, it's also 11.75 MPa at 0.253 strain.
#
# What would this strategy look like in general? I'll plot it below:
#
# Hmm, this isn't a very nice looking result. Let's try something else...

# %%
ax.set_xlim(0.22, 0.28)  # zoom in on 0.22 to 0.28 strain
ax.set_ylim(5, 15)       # zoom in on 5 to 15 MPa stress
fig

print(data[xs][45:55])

idx = np.argmin(np.abs(data[xs] - 0.253))  # find the index of closest x value
print(data.iloc[idx, :])                   # print the corresponding row

xlist = np.linspace(data[xs].min(), data[xs].max(), 1000)
ylist = np.zeros_like(xlist)   # make an array of zeros the same size as xlist
i = 0
for x in xlist:
    idx = np.argmin(np.abs(data[xs] - x))  # find the index of closest x value
    ylist[i] = data[ys][idx]               # assign the corresponding y value
    i += 1
fig2, ax2 = plt.subplots()
ax2.plot(xlist, ylist, c=light_blue)
ax2.set_xlabel(xs)
ax2.set_ylabel(ys + '(interpolated)')
ax2.set_title('Compression of architected material')
