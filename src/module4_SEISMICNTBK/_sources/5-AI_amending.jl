### A Pluto.jl notebook ###
# v0.20.27

using Markdown
using InteractiveUtils

# ╔═╡ f32c7867-492a-4404-8f3d-776d3eac3261
# SegyIO module: for reading data in SEGY format
using SegyIO

# ╔═╡ 59c37877-f77d-40eb-b6e6-9584fc76f500
# Plots module: for visualization
using Plots

# ╔═╡ 83082979-d0d1-47aa-996b-bc1b8408f9fd
# Digital Signal Processing module: for the Hilbert transform
using DSP.Util

# ╔═╡ 686cf9b0-0dd3-4529-8d8e-86637857ba7c
# module Statistics: for quantile function
using Statistics

# ╔═╡ 00e25c16-8165-497e-95a9-4e81fed93e74
# Peaks module: for extracting local maxima
using Peaks

# ╔═╡ d9248fea-e85b-43a4-8ac3-fd2b3cbfab99
# for PCA
using MultivariateStats

# ╔═╡ e6339f81-3bd8-42d9-bfb4-4565fd44a6a0
# for deep learning
using Flux

# ╔═╡ 6a9be9d5-1f4f-4e58-8c4d-5528cfc11cf4
# for shuffling data randomly
using Random

# ╔═╡ 95d65911-1d09-47bd-9c76-1506db129901
using Clustering

# ╔═╡ e111dcb1-d8ff-43fb-ba54-fecf23dfd80e
using ManifoldLearning

# ╔═╡ e60ae28a-6746-4aad-b128-0a75a88d9895
using DataFrames, CSV

# ╔═╡ 0fe8b4f8-0c48-426b-907a-a8afeb6c26bd
using StatisticalMeasures

# ╔═╡ c76e8372-a529-11ef-03ba-a7de23d21592
md"""
# Automating seismic interpretation using artificial intelligence

There are various reasons to attempt to automate seismic interpretation, such as saving labor costs, increasing accuracy, or decreasing the time required for an interpretation project. 

Traditional approaches to automating interpretation involve seismic attributes that highlight important features.
"""

# ╔═╡ b3853692-e4c4-40aa-9c11-647d57de12e0
md"""
![](https://www.researchgate.net/profile/Xinming-Wu-2/publication/327285415/figure/fig1/AS:664950578110473@1535547980484/From-a-3D-seismic-image-a-we-first-compute-an-original-fault-attribute-image-b-from.png)

Wu, X. and Fomel, S., 2018. [Automatic fault interpretation with optimal surface voting](https://library.seg.org/doi/full/10.1190/geo2018-0115.1). Geophysics, 83(5), pp.O67-O82.
"""

# ╔═╡ 29931cf2-a950-4c7b-b2c6-878689d6e5f8
md"""
The recent development of artificial intelligence (AI) and deep learning offered a new approach. Using deep neural networks, AI can automatically generate all the required attributes in the network's hidden layers. It learns data patterns by training on provided data.
"""

# ╔═╡ 84e44b16-e046-4dd9-98e6-ff5863369d87
md"""
![](https://library.seg.org/cms/10.1190/geo2018-0646.1/asset/images/large/figure5.jpeg)

Wu, X., Liang, L., Shi, Y. and Fomel, S., 2019. [FaultSeg3D: Using synthetic data sets to train an end-to-end convolutional neural network for 3D seismic fault segmentation](https://library.seg.org/doi/full/10.1190/geo2018-0646.1). Geophysics, 84(3), pp.IM35-IM45.
"""

# ╔═╡ 14f8891d-fe53-4e5a-93fb-0c80b29631b5
md"""
In this assignment, we will perform some simple deep learning experiments. To learn more about machine learning applications in geoscience problems, including seismic interpretation, you can take the course **GEO398D**, taught in the Spring semester.
"""

# ╔═╡ d6468e7e-2d5a-4973-b1e0-b2a5020966bf
md"""
## Downloading seismic data

We start by downloading the 3D seismic data cube from the Gulf of Mexico dataset.
"""

# ╔═╡ 9488e212-9589-47c5-8999-9382a36d3548
if !isfile("class.sgy")
	download("https://www.dropbox.com/scl/fi/gbdanq178owytm7m3h3tu/class2021.sgy?rlkey=510ivt8ywqul0hx85h8tserma&dl=0", "class.sgy")
end

# ╔═╡ 244906fc-ade7-41c1-aa39-260fa541073d
md"""
!!! warning

    The file `class.sgy` is large (5.4 Gb) and make take some time to download.
"""

# ╔═╡ 6e2db287-84a2-4ecb-bc58-55a975b2cd1f
seismic = segy_read("class.sgy");

# ╔═╡ 770a98ad-782d-4998-8cb0-cc9831014869
(nt, ntraces) = size(seismic.data)

# ╔═╡ 48c069f9-24ae-4dc3-8aa5-ff460ca3be81
# Get inline and crossline information
begin
	iline = get_header(seismic,:Inline3D)
	xline = get_header(seismic,:Crossline3D)
	il = range(start=minimum(iline), stop=maximum(iline))
	xl = range(start=minimum(xline), stop=maximum(xline))
	ilines, xlines = length(il), length(xl)
	xlines * ilines == ntraces
end

# ╔═╡ 670e6291-f5a8-41ea-b342-2c270110909a
begin
	# time sampling
	dt = get_header(seismic, :dt)[1] * 1e-6
	time = range(start=0, length=nt, step=dt)
end

# ╔═╡ ec324d61-bd97-4ac6-beb9-7c6e1aaa2213
# create a three-dimensional cube and convert data to standard Float32
cube = reshape(Float32.(seismic.data)/1000, (nt, xlines, ilines));

# ╔═╡ c423f3e0-846c-4148-a45f-551654a6b471
md"""
## Unsupervised learning

We will first try to learn patterns directly from the data using *unsupervised* learning.

For simplicity, let us select one vertical section from the data cube.
"""

# ╔═╡ 499ba1e1-ec34-4ef7-ab9e-bf1c12a489c1
section = cube[:,100,:];

# ╔═╡ be428b20-0d19-4e17-be89-f69029c10285
heatmap(il, time[1:1001], section[1:1001,:], 
	    yflip=:true, cmap=:grays, clim=(-5, 5),
	    xlabel="Inline", ylabel="Time (s)", title="Crossline Section")

# ╔═╡ 1c32cd0c-30be-4cbb-95ce-1676aea650f6
md"""
We will break the section into a collection of seismic wavelets (small 1-D windows) and analyze the differences and similarities in their patterns.

To have one common feature among the wavelets, we will put their centers at the local maxima of the seismic energy, as measured by the envelope.
"""

# ╔═╡ 858b9411-356e-443b-8a54-238a5899dd20
# apply the Hilbert transform on each trace in the section 
envelope = abs.(mapslices(hilbert, section, dims=1));

# ╔═╡ f04e3f8d-5e5b-4f69-886e-2307fa4bb7e5
heatmap(il, time[1:1001], envelope[1:1001,:], yflip=:true, clim=(0, 5),
	xlabel="Inline", ylabel="Time (s)", title="Envelope")

# ╔═╡ f81462e4-a76a-4f2c-9389-bb9b588c978d
# hard thresholding
function threshold(data, percent)
	threshold = quantile(data[:], percent/100)
	data_t = similar(data)
	for k in 1:length(data)
		if data[k] < threshold
			data_t[k] = zero(Float32)
		else
			data_t[k] = data[k]
		end
	end
	return data_t
end

# ╔═╡ a88282a1-e3d4-4fa6-8080-ec72e45d6030
envelope_t = threshold(envelope, 80);

# ╔═╡ dcbf426d-efb8-47c7-ba54-9759a17959cc
heatmap(il, time[1:1001], envelope_t[1:1001,:], yflip=:true, clim=(0, 5),
	xlabel="Crossline", ylabel="Time (s)", title="Thresholded Envelope")

# ╔═╡ 11f00f03-0278-4e4e-ae7c-7ef65c81d3b9
function extract_wavelets(seis, env; nw = 41)
	nt, ntraces = size(seis)
	wavelets = Array{Vector{Float32}}(undef,0)
	picks = Array{Vector{Float32}}(undef,0)
	for n in 1:ntraces
    	trace = seis[:, n]
    	etrace = env[:, n]
		# local maxima of envelope
	    peaks, = findmaxima(etrace)
    	for peak in peaks
        	if peak > nw && peak < nt-nw
				# wavelet centered on a local maximum
            	wavelet = trace[peak-nw÷2:peak+nw÷2]
            	if wavelet[1+nw÷2] == maximum(wavelet) 
					push!(wavelets, wavelet/maximum(abs.(wavelet)))
					push!(picks, [il[1]+n-1, (peak-1)*dt])
				end
			end
		end
	end
	return picks, wavelets
end

# ╔═╡ e3a06b53-fd6d-45fa-8e34-15e6bf29fdca
nw = 41 # wavelet length

# ╔═╡ 6875bc49-bfd5-491f-b4c0-7fd491e18c2a
picks, wavelets = extract_wavelets(section, envelope_t, nw=nw);

# ╔═╡ ae2b7102-7c30-49e7-897b-0537085f24dd
# put all wavelets in a matrix
Mwavelets = reduce(hcat, wavelets);

# ╔═╡ c8375cfe-5f90-49af-9ebb-6eadd70ac655
size(Mwavelets)

# ╔═╡ c4b1290d-ca33-44ec-809d-2191a08ff206
md"""
At this step, we collected around six thousand wavelets containing 41 samples. Let's plot some of them.
"""

# ╔═╡ cd7d2586-aa65-4aff-bc5b-f5ab5b5663d8
function plot_wavelets(wavelets, traces, title)
    nw = size(wavelets, 1)
	twt = (-nw÷2:nw÷2)*dt
    ncols = length(traces)
	plt = Array{Plots.Plot}(undef,ncols)
    for n in 1:ncols
        # plot as curve
        amp = wavelets[:,traces[n]]
		if n==1
			plt[n] = plot(amp, twt, color=:black, label=:none, 
				          ylabel="Time (s)", xaxis = false)
		else
			plt[n] = plot(amp, twt, color=:black, label=:none, showaxis = false)
		end
	end
    return plot(plt..., layout=(1, ncols), plot_title=title)
end

# ╔═╡ 72f2eda2-5d34-4e32-9659-0d8b50bdd126
plot_wavelets(Mwavelets, 100:100:600, "Example Wavelets")

# ╔═╡ 3ffb9f00-f919-4d21-bfc0-3132538e9920
md"""
### Reducing size

Our wavelets exist in 41-dimensional space. However, not all of the little wiggles are important for comparison. Compressing the information helps in extracting common patterns.

A classic dimensionality reduction method is **PCA** (Principal Component Analysis) and is based on the technique of **SVD** (Singular Value Decomposition) from numerical linear algebra.
"""

# ╔═╡ 2b1f908e-bf92-4b04-888d-f82539355298
md"""
![](https://upload.wikimedia.org/wikipedia/commons/thumb/c/c8/Singular_value_decomposition_visualisation.svg/512px-Singular_value_decomposition_visualisation.svg.png)

* [https://en.wikipedia.org/wiki/Singular\_value\_decomposition](https://en.wikipedia.org/wiki/Singular_value_decomposition)
"""

# ╔═╡ cf999cbe-0711-41c4-99a0-d4cbb9c5cd8f
pca = fit(PCA, Mwavelets)

# ╔═╡ e391d5a2-825b-4b74-856d-13294a7dc14f
bar(eigvals(pca), label=:none, title="PCA Singular Values") 

# ╔═╡ a288fc10-bcd8-4a7a-8aa4-f07f876e6bea
md"""
A quick decay in singular values shows a possibility of compression.
"""

# ╔═╡ fb34a3e6-f84b-4454-8cb9-6d625bdbe643
plot_wavelets(projection(pca), 1:6, "Principal Components")

# ╔═╡ bd10b539-dc98-42ca-aecd-df583377a3f0
# compress to size 10
pca10 = fit(PCA, Mwavelets; maxoutdim=10);

# ╔═╡ 6616912c-576d-4ad2-b754-ea4a3e65ecb6
# latent space
latent = MultivariateStats.transform(pca10, Mwavelets);

# ╔═╡ b113f04e-67c2-41a8-8857-2cc466415f44
size(latent)

# ╔═╡ 4c1c5a5c-cf4e-4b59-8e5c-fb3a9a71b17e
# reconstruction
pca_wavelets = reconstruct(pca10, latent);

# ╔═╡ ba99833b-947a-4ca4-8142-ea928f3f5772
plot_wavelets(pca_wavelets, 100:100:600, "PCA Wavelets")

# ╔═╡ 21b63603-d840-48ea-a4da-4904e5f37678
plot_wavelets(Mwavelets, 100:100:600, "Example Wavelets")

# ╔═╡ 06b85900-2034-44fc-856c-da440e8b87a7
md"""
### Autoencoder

Deep learning provides a more robust method of dimensionality reduction. While PCA uses linear operations (matrix multiplications), artificial neural networks create chains of operators, where linear operations are sandwiched with nonlinear mappings. This construction, inspired by the functioning of neurons in a human brain, can approximate complex relationships. It will help us better extract patterns from seismic wavelets.
"""

# ╔═╡ c02c1652-57d6-4089-8f34-a37e7072049c
function create_autoencoder(n_hidden, n_latent)
	encoder = Chain(Dense(nw => n_hidden, relu), Dense(n_hidden => n_latent, relu))
	decoder = Chain(Dense(n_latent => n_hidden, relu), Dense(n_hidden => nw, tanh))
	return Chain(encoder, decoder)
end

# ╔═╡ 5b17ee99-c946-408c-a94a-41fb63a74f51
md"""
Our deep learning model consists of an encoder and a decoder. The encoder transforms the input wavelets to a latent lower-dimensional space while the decoder reconstructs the original dimensions. Both are chains of linear operations followed by activation functions.
"""

# ╔═╡ 49d698d5-d37c-4e9b-8fc1-402a263fb69c
md"""
![](https://upload.wikimedia.org/wikipedia/commons/thumb/4/46/Colored_neural_network.svg/638px-Colored_neural_network.svg.png)

* [https://en.wikipedia.org/wiki/Neural\_network\_\(machine\_learning\)](https://en.wikipedia.org/wiki/Neural_network_%28machine_learning%29)
"""

# ╔═╡ 2ef52597-7ecd-4946-9bc6-505f989faa55
md"""
In a mathematical notation, we can express the encoder as

$\mathbf{y}(\mathbf{x}) = r\left(\mathbf{W}_2\,r\left(\mathbf{W}_1\,\mathbf{x}+\mathbf{b}_1\right) + \mathbf{b}_2\right)\;,$

where $\mathbf{x}$ is the input, $\mathbf{y}$ is the output, $\mathbf{W}_1$, $\mathbf{b}_1$, $\mathbf{W}_2$, $\mathbf{b}_2$ are model parameters (to be estimated by training), and $r$ stands for the ReLU (rectified linear unit) activation function.
"""

# ╔═╡ 1a750016-ff4e-4575-8764-39568c91a5db
plot(relu, label=:none, title="ReLU activation function")

# ╔═╡ a9a597fa-9176-45e0-b494-bc00a0005c98
md"""
Correspondingly, the decoder is

$\mathbf{x}(\mathbf{y}) = t\left(\mathbf{W}_4\,r\left(\mathbf{W}_3\,\mathbf{y}+\mathbf{b}_3\right) + \mathbf{b}_4\right)\;,$

In this case, the output activation function $t$ is the hyperbolic tangent, which guarantees that the output values will be between -1 and 1.
"""

# ╔═╡ 2160f090-5040-47fa-9ae4-9d2fb6391c96
plot(tanh, label=:none, title="Tanh activation function")

# ╔═╡ 200f314f-dae0-43be-82fa-b609a061142b
md"""
### Training the autoencoder

To test the model's effectiveness, we will separate the input data into training and testing sets, the latter will validate accuracy.
"""

# ╔═╡ a9291696-b531-499e-8c96-c070e937fcf0
function test_train_split(data, perc)
	Random.seed!(2024) # for reproducibility
    n = size(data, 2)
	split = floor(Int, perc*n)
	idx = shuffle(1:n)
    train_idx = view(idx, 1:split)
    test_idx = view(idx, split+1:n)
    return data[:, train_idx], data[:, test_idx]
end

# ╔═╡ a5f54d85-85e6-4060-970b-b8c93fececfb
Mtrain, Mtest = test_train_split(Mwavelets, 0.8);

# ╔═╡ d34b12b6-e777-4311-9cc6-3f83124e57b9
size(Mtrain), size(Mtest)

# ╔═╡ 7d1954f6-4954-4e48-a934-9e87d0d29363
function train(train, test; epochs=10, batchsize=16, n_hidden=21, n_latent=11)
	train_loss = Array{Float32}(undef, epochs)
	test_loss = Array{Float32}(undef, epochs)
	model = create_autoencoder(n_hidden, n_latent)
	state = Flux.setup(Adam(), model) # set the optimization method
	loader = Flux.DataLoader((train, train), batchsize=batchsize)	
	for epoch in 1:epochs
		# loop over batches of training data
	    for (x, _) in loader
    	    # Compute loss and gradient
        	l, gs = Flux.withgradient(m -> Flux.mae(m(x), x), model)
        	# Update model parameters
        	Flux.update!(state, model, gs[1])
    	end
		train_loss[epoch] = Flux.mae(model(train), train)
		test_loss[epoch] = Flux.mae(model(test), test)
		println("$epoch train_loss=$(train_loss[epoch]) test_loss=$(test_loss[epoch])")
	end
	return model, train_loss, test_loss
end

# ╔═╡ 7ad5b84b-02cd-4a74-9888-072be9366770
plot_wavelets(Mwavelets, 100:100:600, "Example Wavelets")

# ╔═╡ 770026ee-d572-40a0-a841-52f00cd27f75
md"""
### Clustering

After transforming the input wavelets into a lower-dimensional latent space, we can perform a clustering operation to group them.
"""

# ╔═╡ 8c965cbf-9b46-4d36-a7b3-11d4a82e0b72
md"""
K-means is a popular clustering algorithm. It works by looking for k centers (mean points) in the space of the clustered objects. Each object is then assigned to the nearest center point.

The algorithm works iteratively:

1. Start with some distribution of centers, possibly random.
2. Divide the data into clusters and compute the mean point for each cluster.
3. If the mean point of a cluster does not coincide with its center, move the center.
4. Repeat step 2.

* [https://en.wikipedia.org/wiki/K-means\_clustering](https://en.wikipedia.org/wiki/K-means_clustering)
"""

# ╔═╡ ea9d599d-db36-40ec-a958-e1972cb2f475
md"""
!!! assignment
    ### Task 1

    Change the number of clusters.
"""

# ╔═╡ d5b8ce01-3dfa-480a-8bd2-7d90c88e798a
md"""
How do we find an optimal number of clusters (the $k$ in k-Means) for a given dataset?

One popular approach is the "elbow method." If we compute the cost of the k-Means cluster, also called *inertia* (the sum of the squared distances between each data point and its nearest cluster center), it will decrease with $k$. However, there can be a characteristic point (an "elbow") where the rate of the decrease changes. Empirically, the value of $k$ at this point is a good choice for the number of clusters.

* [https://en.wikipedia.org/wiki/Elbow\_method\_(clustering)](https://en.wikipedia.org/wiki/Elbow_method_(clustering))
"""

# ╔═╡ 46e0ba73-31aa-4ae3-94d5-f988f51318d1
function elbow_method(data, ks)
	nk = length(ks)
	inertia = Array{Float64}(undef, nk)
	for ik in 1:nk
		km = kmeans(data, ks[ik])
	    inertia[ik] = km.totalcost
	end
	return inertia
end

# ╔═╡ cf8b5fde-cba4-4f19-b14e-2fe12254a041
md"""
**Your task**: Plot the inertia, find the elbow point, and change the number of clusters above if the elbow is not at $k=4$.
"""

# ╔═╡ 54d5ae08-7443-4dfc-8859-e4233e856712
md"""
### Visualizing clusters using t-SNE

**T-SNE** (t-distributed Stochastic Neighbor Embedding) is a technique for visualizing objects from a multidimensional space by projecting them into two or three dimensions.

* [https://en.wikipedia.org/wiki/T-distributed\_stochastic\_neighbor\_embedding](https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding)
"""

# ╔═╡ d93b71e4-5882-4713-868e-241adb0681da
md"""
## Supervised Learning

In supervised learning, an AI model aims to approximate mapping from the input to the desired output, using many instances of input-output pairs for training. 

Provided a manually picked horizon, we will apply it to train a model for classifying seismic wavelets.
"""

# ╔═╡ 4d9cf901-a42f-4807-92a3-cc183965c553
md"""
### Downloading a picked horizon

We start by downloading a seafloor horizon.
"""

# ╔═╡ e7c4dea7-2c2a-496f-97a1-96c045a4a6e5
download("https://raw.githubusercontent.com/zsylvester/GEO391_materials/master/FE_horizons/seafloor_XY_il_cl.csv","seafloor.csv")

# ╔═╡ 4d9ea67f-6948-4e3b-9b0f-81c9207f65d5
df = DataFrame(CSV.File("seafloor.csv"), ["x", "y", "il", "xl", "z"])

# ╔═╡ 684eba81-0212-4943-a2dc-2ef28ebff516
begin
	ilmin, ilmax  = minimum(df.il), maximum(df.il);
	xlmin, xlmax  = minimum(df.xl), maximum(df.xl);
end

# ╔═╡ a5726855-2e54-44ea-abfc-5b0008159ef7
function df2horizon(df,nil,nxl,ilmin,xlmin)
	# extract a horizon from a data frame
	z = zeros(nxl, nil)
	for i in 1:nrow(df)
		z[df.xl[i] - xlmin + 1, df.il[i] - ilmin + 1] = df.z[i]/1000
	end
	return z
end

# ╔═╡ ca6e34ce-4531-468c-8763-540be2b00ee7
sf = df2horizon(df, ilmax-ilmin+1, xlmax-xlmin+1, ilmin, xlmin);

# ╔═╡ 29ae22fd-4582-4c9c-829c-462d8dfb0286
begin
	heatmap(il, time[1:1001], section[1:1001,:], legend=:none, 
		    yflip=:true, cmap=:grays, clim=(-5, 5),
		    xlabel="Inline", ylabel="Time (s)", title="Seismic + Horizon")
	scatter!(il, sf[100,:], color=:yellow, 
		    xlim=(il[1], il[ilines]), ylim=(0, 4), 
		    label=:none, yflip=true, markersize=1, markerstrokewidth=0)
end

# ╔═╡ 7418f53f-d948-4e18-b485-2213ee8dca9a
md"""
### Labeling seafloor wavelets

We can assign labels to different seismic wavelets using the picked horizon: 1 for wavelets centered on the seafloor and 0 for all other wavelets.
"""

# ╔═╡ 7fef9d14-d63e-479d-a2cc-04b3194dd883
function label_wavelets(picks, seafloor)
	npicks = length(picks)
	label = Array{Int}(undef,npicks)
	for k in 1:npicks
		pick = picks[k]
		n = floor(Int, pick[1]-il[1]+1)
		if abs(seafloor[n] - pick[2]) < dt
			label[k] = 1
		else
			label[k] = 0
		end
	end
	return label
end

# ╔═╡ 74128555-0120-4eae-9bed-38d8e6fd58c8
label = label_wavelets(picks, sf[100,:])

# ╔═╡ e153520d-592a-4d20-bef3-879a532a8438
md"""
### Deep-learning classification.

The classification model is analogous to the encoder model we used previously. We will use two hidden layers. The output will contain only one point, representing the probability of an input wavelet being a seafloor wavelet. To ensure this value is between 0 and 1, we will use the sigmoid activation function $\sigma$.
"""

# ╔═╡ 57144295-0516-4642-8066-0872840a0fca
function create_classifier(n_hidden1, n_hidden2)
	return Chain(
		Dense(nw => n_hidden1, relu), 
		Dense(n_hidden1 => n_hidden2, relu),
		Dense(n_hidden2 => 1, σ)
	)
end

# ╔═╡ c2bec94f-7b7d-4b90-ba3b-1e62b36a9922
plot(σ, label=:none, title="Sigmoid activation function")

# ╔═╡ 29b05235-d45f-4919-942e-801ae4895876
md"""
In a mathematical notation, our classifier model is

$p(\mathbf{x}) =  \sigma\left(\mathbf{w}_3^T\,r\left(\mathbf{W}_2\,\left(\mathbf{W}_1\,\mathbf{y}+\mathbf{b}_1\right)+\mathbf{b}_2\right) + b_3\right)\;.$

When the output probability $p$ is mapped to the nearest value of 0 or 1, it can indicate whether the given wavelet $\mathbf{x}$ is a seafloor wavelet.
"""

# ╔═╡ 48610899-63d5-4dd5-8836-80afb4f2bc89
# split labels for training nd testing
ytrain, ytest = test_train_split(hcat(label)', 0.8);

# ╔═╡ 837cd1f4-72d0-418e-9f45-900862e6987f
function train(xtrain, ytrain, xtest, ytest; 
               epochs=10, batchsize=16, n_hidden1=11, n_hidden2=6)
	train_loss = Array{Float32}(undef, epochs)
	test_loss = Array{Float32}(undef, epochs)
	model = create_classifier(n_hidden1, n_hidden2)
	state = Flux.setup(Adam(), model)
	loader = Flux.DataLoader((xtrain, ytrain), batchsize=batchsize)	
	for epoch in 1:epochs
		# loop over batches of training data
	    for (x, y) in loader
    	    # Compute loss and gradient
        	l, gs = Flux.withgradient(m -> Flux.binarycrossentropy(m(x), y), model)
        	# Update model parameters
        	Flux.update!(state, model, gs[1])
    	end
		train_loss[epoch] = Flux.binarycrossentropy(model(xtrain), ytrain)
		test_loss[epoch] = Flux.binarycrossentropy(model(xtest), ytest)
		println("$epoch train_loss=$(train_loss[epoch]) test_loss=$(test_loss[epoch])")
	end
	return model, train_loss, test_loss
end

# ╔═╡ 65c6c85a-a4cd-4f75-a5d4-dae16b4aabde
autoencoder, train_loss, test_loss = train(Mtrain, Mtest, epochs=50);

# ╔═╡ 144027d3-d6fe-4a79-b1b8-9a5ebe4af6ee
plot([train_loss test_loss], label=["train loss" "test loss"], 
     xlabel="Epoch", ylabel="Loss")

# ╔═╡ 5108bf24-6235-4e94-9798-c954e8ae2416
encoder, decoder = autoencoder.layers;

# ╔═╡ fd8b3573-48e4-4778-8d17-b2bf1971cf03
encoded = encoder(Mwavelets);

# ╔═╡ 4eacd3df-ab32-481f-bce5-a9c5576d6342
size(encoded)

# ╔═╡ e543aadf-cbb4-4635-b8bf-e57241af1d8c
cluster = kmeans(encoded, 4);

# ╔═╡ 4290f11d-d8c4-4a71-8b38-b1d34802818c
nclusters(cluster)

# ╔═╡ fe24979c-ae83-412b-a25e-2e3c3fc19c1f
begin
	heatmap(il, time[1:1001], section[1:1001,:], legend=:none, 
		    yflip=:true, cmap=:grays, clim=(-5, 5),
		    xlabel="Inline", ylabel="Time (s)")
	scatter!(Tuple.(picks), color=cluster.assignments, 
		    xlim=(il[1], il[ilines]), ylim=(0, 4), 
		    label=:none, yflip=true, markersize=3, markerstrokewidth=0)
end

# ╔═╡ 7351fb16-79b9-4a48-9c11-785ec38d034c
inertia = elbow_method(encoded, 1:10)

# ╔═╡ e5e4c45d-b9a9-46fe-8064-61231ca41b45
manifold = fit(TSNE, encoded)

# ╔═╡ 1f31de5e-51e2-4e0e-85d4-3f3c1a6196e9
reduction = ManifoldLearning.predict(manifold);   

# ╔═╡ b1006a0b-d3f1-4f20-9bb9-0f82f111c209
size(reduction)

# ╔═╡ 9e8ab52c-bff7-4250-8a05-8d2f3b7a4c80
scatter(reduction[1,:], reduction[2,:], title="t-SNE Projection",
	    color=cluster.assignments, label=:none, 
	    xlabel="Latent 1", ylabel="Latent 2")

# ╔═╡ da39a0d6-c21f-4a43-a8a8-b01761db9886
decoded = decoder(encoded);

# ╔═╡ 750e2715-b8f0-4313-97ec-246223cd5c9e
size(decoded)

# ╔═╡ aefb8469-5386-45bf-8a37-901c5efb60e6
plot_wavelets(decoded, 100:100:600, "Autoencoder Wavelets")

# ╔═╡ 75cfbd96-be0f-4697-93e5-cf84ec76e36d
size(Mtrain), size(ytrain), size(Mtest), size(ytest)

# ╔═╡ 0dea764e-fbfa-4929-8ce5-42f0ba0cbee5
classifier, train_loss2, test_loss2 = train(Mtrain, ytrain, Mtest, ytest, epochs=50);

# ╔═╡ 0c35b060-bfd4-4187-87eb-91ebae01f194
plot([train_loss2 test_loss2], label=["train loss" "test loss"], 
     xlabel="Epoch", ylabel="Loss")

# ╔═╡ 1f39ab61-7d2d-4b76-9f15-9ed0074a2339
classifier(Mwavelets)

# ╔═╡ e39e90a8-7c95-4595-92c3-79d5d206670d
predict = round.(Int,classifier(Mwavelets))

# ╔═╡ ae51631e-9a36-4023-90d4-e85524ebb92d
md"""
To evaluate the accuracy of the prediction, it helps to look at the *confusion matrix*, which divides all outcomes into true positives, true negatives, false positives, and false negatives.
"""

# ╔═╡ 80ae48ad-21bd-45c5-abcd-971e48a09d44
confmat(predict, label)

# ╔═╡ 65ff6161-4280-4b92-befd-0e23cd3e701b
begin
	heatmap(il, time[1:1001], section[1:1001,:], legend=:none, 
		    yflip=:true, cmap=:grays, clim=(-5, 5),
		    xlabel="Inline", ylabel="Time (s)")
	scatter!(Tuple.(picks[predict[1,:] .== 1]), color=:yellow, 
		    xlim=(il[1], il[ilines]), ylim=(0, 4), 
		    label=:none, yflip=true, markersize=1, markerstrokewidth=0)
end

# ╔═╡ 3b822a24-cccb-46e4-a60b-9b8a01b358d7
md"""
### Testing the classifier on a different section

Now that we have a trained wavelet classifier, we can apply it to identify the seafloor at other sections. 
"""

# ╔═╡ 64d046ac-3388-4862-835f-78437096b3f6
section2 = cube[:,120,:];

# ╔═╡ c8cc8431-f4e3-46fa-a293-32dd8479782f
envelope2 = abs.(mapslices(hilbert, section2, dims=1));

# ╔═╡ 1e86c61c-a2b4-4ec7-951d-a52132f96fa6
envelope2_t = threshold(envelope2, 80);

# ╔═╡ 31a28fe4-4c9b-40cf-ada1-c04a8955914f
picks2, wavelets2 = extract_wavelets(section2, envelope2_t, nw=nw);

# ╔═╡ 360f1f4d-d8a4-4502-a8b7-27f0a018da33
# put wavelets in a matrix
Mwavelets2 = reduce(hcat, wavelets2);

# ╔═╡ 0cec4057-ad39-443e-86e7-d1feebc73278
predict2 = round.(Int, classifier(Mwavelets2));

# ╔═╡ 54edc22a-99ee-4b97-902a-5c1eff374938
begin
	heatmap(il, time[1:1001], section2[1:1001,:], legend=:none, 
		    yflip=:true, cmap=:grays, clim=(-5, 5),
		    xlabel="Inline", ylabel="Time (s)")
	scatter!(Tuple.(picks2[predict2[1,:] .== 1]), color=:yellow, 
		    xlim=(il[1], il[ilines]), ylim=(0, 4), 
		    label=:none, yflip=true, markersize=1, markerstrokewidth=0)
end

# ╔═╡ 05671fec-1cdd-43cc-bd5d-cc1231e9249a
md"""
Real-life applications of AI in seismic data interpretation often use a similar approach: turning manual interpretation into labels for training a deep-learning model through supervised learning and then applying the model to other parts of the data to automate the process.
"""

# ╔═╡ a48200da-05e4-40ad-aa5b-6622da4e1ade
md"""
!!! assignment
    ### Task 2

    Apply supervised learning to detect a different horizon.
"""

# ╔═╡ 910b28ec-b2d4-4a71-9168-acf221b88b5d
md"""
Let us try a different manually picked horizon from the same data.
"""

# ╔═╡ 773af3b1-ee4a-4e02-9a46-cf1727567188
download("https://raw.githubusercontent.com/zsylvester/GEO391_materials/master/FE_horizons/horizon_03.csv","horizon.csv")

# ╔═╡ b0b2201f-0193-42c2-99f6-571649051b9f
df2 = DataFrame(CSV.File("horizon.csv"), ["il", "xl", "x", "y", "z"])

# ╔═╡ e2fabd8b-c499-4316-aafc-cfffea951c38
horizon = df2horizon(df2, ilmax-ilmin+1, xlmax-xlmin+1, ilmin, xlmin);

# ╔═╡ d0c67b7c-e341-4ec6-85ae-fe46240b632a
begin
	heatmap(il, time[1:1001], section[1:1001,:], legend=:none, 
		    yflip=:true, cmap=:grays, clim=(-5, 5),
		    xlabel="Inline", ylabel="Time (s)", title="Seismic + Horizon")
	scatter!(il, horizon[100,:], color=:cyan, 
		    xlim=(il[1], il[ilines]), ylim=(0, 4), 
		    label=:none, yflip=true, markersize=1, markerstrokewidth=0)
end

# ╔═╡ 7d350757-15fb-43db-b9ec-2c23a28bad34
label2 = label_wavelets(picks, horizon[100,:])

# ╔═╡ b02f73fb-7ec3-4345-a585-252de5fedc92
sum(label2)

# ╔═╡ d2934768-fa38-4404-b69e-f01d3bd6f39c
md"""
There are 166 wavelets from the second horizon in our collection.

**Your task**: Create and train a classification model for detecting this kind of wavelet and test its accuracy. 
"""

# ╔═╡ 94b7ba46-ce14-460f-a741-6059db8755c4
md"""
!!! assignment
    ### Bonus tasks

    For extra credit, you can work on the following tasks:

    1. Use a horizon that you picked yourself instead of the ones provided above.
    2. Implement a multiclass classification model that detects multiple horizons simultaneously.
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
Clustering = "aaaa29a8-35af-508c-8bc3-b662a17a0fe5"
DSP = "717857b8-e6f2-59f4-9121-6e50c889abd2"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
Flux = "587475ba-b771-5e3f-ad9e-33799f191a9c"
ManifoldLearning = "06eb3307-b2af-5a2a-abea-d33192699d32"
MultivariateStats = "6f286f6a-111f-5878-ab1e-185364afe411"
Peaks = "18e31ff7-3703-566c-8e60-38913d67486b"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
SegyIO = "157a0f19-4d44-4de5-a0d0-07e2f0ac4dfa"
StatisticalMeasures = "a19d573c-0a75-4610-95b3-7071388c7541"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[compat]
CSV = "~0.10.16"
Clustering = "~0.15.8"
DSP = "~0.8.4"
DataFrames = "~1.8.2"
Flux = "~0.14.25"
ManifoldLearning = "~0.9.0"
MultivariateStats = "~0.9.1"
Peaks = "~0.5.3"
Plots = "~1.41.6"
SegyIO = "~0.8.5"
StatisticalMeasures = "~0.1.7"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.6"
manifest_format = "2.0"
project_hash = "64927ab533e3f55fd8180ce2aef48fcc87d11bb3"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "d92ad398961a3ed262d8bf04a1a2b8340f915fef"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.5.0"
weakdeps = ["ChainRulesCore", "Test"]

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"
    AbstractFFTsTestExt = "Test"

[[deps.Accessors]]
deps = ["CompositionsBase", "ConstructionBase", "Dates", "InverseFunctions", "MacroTools"]
git-tree-sha1 = "856ecd7cebb68e5fc87abecd2326ad59f0f911f3"
uuid = "7d9f7c33-5ae7-4f3b-8dc6-eff91059b697"
version = "0.1.43"

    [deps.Accessors.extensions]
    AxisKeysExt = "AxisKeys"
    IntervalSetsExt = "IntervalSets"
    LinearAlgebraExt = "LinearAlgebra"
    StaticArraysExt = "StaticArrays"
    StructArraysExt = "StructArrays"
    TestExt = "Test"
    UnitfulExt = "Unitful"

    [deps.Accessors.weakdeps]
    AxisKeys = "94b1ba4f-4ee9-5380-92f1-94cde586c3c5"
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "0761717147821d696c9470a7a86364b2fbd22fd8"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.5.2"
weakdeps = ["SparseArrays", "StaticArrays"]

    [deps.Adapt.extensions]
    AdaptSparseArraysExt = "SparseArrays"
    AdaptStaticArraysExt = "StaticArrays"

[[deps.ArgCheck]]
git-tree-sha1 = "f9e9a66c9b7be1ad7372bbd9b062d9230c30c5ce"
uuid = "dce04be8-c92d-5529-be00-80e4d2c0e197"
version = "2.5.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.ArnoldiMethod]]
deps = ["LinearAlgebra", "Random", "StaticArrays"]
git-tree-sha1 = "d57bd3762d308bded22c3b82d033bff85f6195c6"
uuid = "ec485272-7323-5ecc-a04f-4719b315124d"
version = "0.4.0"

[[deps.Arpack]]
deps = ["Arpack_jll", "Libdl", "LinearAlgebra", "Logging"]
git-tree-sha1 = "9b9b347613394885fd1c8c7729bfc60528faa436"
uuid = "7d9fca2a-8960-54d3-9f78-7d1dccf2cb97"
version = "0.5.4"

[[deps.Arpack_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "OpenBLAS_jll", "Pkg"]
git-tree-sha1 = "5ba6c757e8feccf03a1554dfaf3e26b3cfc7fd5e"
uuid = "68821587-b530-5797-8361-c406ea357684"
version = "3.5.1+1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Atomix]]
deps = ["UnsafeAtomics"]
git-tree-sha1 = "b8651b2eb5796a386b0398a20b519a6a6150f75c"
uuid = "a9b6321e-bd34-4604-b9c9-b65b8de01458"
version = "1.1.3"

    [deps.Atomix.extensions]
    AtomixCUDAExt = "CUDA"
    AtomixMetalExt = "Metal"
    AtomixOpenCLExt = "OpenCL"
    AtomixoneAPIExt = "oneAPI"

    [deps.Atomix.weakdeps]
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    Metal = "dde4c033-4e86-420c-a63e-0dd931031962"
    OpenCL = "08131aa3-fb12-5dee-8b74-c09406e224a2"
    oneAPI = "8f75cd03-7ff8-4ecb-9b8f-daf728133b1b"

[[deps.BangBang]]
deps = ["Accessors", "ConstructionBase", "InitialValues", "LinearAlgebra"]
git-tree-sha1 = "26f41e1df02c330c4fa1e98d4aa2168fdafc9b1f"
uuid = "198e06fe-97b7-11e9-32a5-e1d131e6ad66"
version = "0.4.4"

    [deps.BangBang.extensions]
    BangBangChainRulesCoreExt = "ChainRulesCore"
    BangBangDataFramesExt = "DataFrames"
    BangBangStaticArraysExt = "StaticArrays"
    BangBangStructArraysExt = "StructArrays"
    BangBangTablesExt = "Tables"
    BangBangTypedTablesExt = "TypedTables"

    [deps.BangBang.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Tables = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
    TypedTables = "9d95f2ec-7b3d-5a63-8d20-e2491e220bb9"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.Baselet]]
git-tree-sha1 = "aebf55e6d7795e02ca500a689d326ac979aaf89e"
uuid = "9718e550-a3fa-408a-8086-8db961cd8217"
version = "0.1.1"

[[deps.Bessels]]
git-tree-sha1 = "4435559dc39793d53a9e3d278e185e920b4619ef"
uuid = "0e736298-9ec6-45e8-9647-e4fc86a2fe38"
version = "0.2.8"

[[deps.BitFlags]]
git-tree-sha1 = "bbe1079eecf9c9fbb52765193ad2bae27ae09bc8"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.10"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1b96ea4a01afe0ea4090c5c8039690672dd13f2e"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.9+0"

[[deps.CEnum]]
git-tree-sha1 = "389ad5c84de1ae7cf0e28e381131c98ea87d54fc"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.5.0"

[[deps.CSV]]
deps = ["CodecZlib", "Dates", "FilePathsBase", "InlineStrings", "Mmap", "Parsers", "PooledArrays", "PrecompileTools", "SentinelArrays", "Tables", "Unicode", "WeakRefStrings", "WorkerUtilities"]
git-tree-sha1 = "8d8e0b0f350b8e1c91420b5e64e5de774c2f0f4d"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "0.10.16"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "1fa950ebc3e37eccd51c6a8fe1f92f7d86263522"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.7+0"

[[deps.CategoricalArrays]]
deps = ["DataAPI", "Future", "Missings", "Printf", "Requires", "Statistics", "Unicode"]
git-tree-sha1 = "1568b28f91293458345dabba6a5ea3f183250a61"
uuid = "324d7699-5711-5eae-9e2f-1d82baa6b597"
version = "0.10.8"

    [deps.CategoricalArrays.extensions]
    CategoricalArraysJSONExt = "JSON"
    CategoricalArraysRecipesBaseExt = "RecipesBase"
    CategoricalArraysSentinelArraysExt = "SentinelArrays"
    CategoricalArraysStructTypesExt = "StructTypes"

    [deps.CategoricalArrays.weakdeps]
    JSON = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
    RecipesBase = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
    SentinelArrays = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
    StructTypes = "856f2bd8-1eba-4b0a-8007-ebc267875bd4"

[[deps.CategoricalDistributions]]
deps = ["CategoricalArrays", "Distributions", "Missings", "OrderedCollections", "Random", "ScientificTypes"]
git-tree-sha1 = "926862f549a82d6c3a7145bc7f1adff2a91a39f0"
uuid = "af321ab8-2d2e-40a6-b165-3d674595d28e"
version = "0.1.15"

    [deps.CategoricalDistributions.extensions]
    UnivariateFiniteDisplayExt = "UnicodePlots"

    [deps.CategoricalDistributions.weakdeps]
    UnicodePlots = "b8865327-cd53-5732-bb35-84acbb429228"

[[deps.ChainRules]]
deps = ["Adapt", "ChainRulesCore", "Compat", "Distributed", "GPUArraysCore", "IrrationalConstants", "LinearAlgebra", "Random", "RealDot", "SparseArrays", "SparseInverseSubset", "Statistics", "StructArrays", "SuiteSparse"]
git-tree-sha1 = "3c190c570fb3108c09f838607386d10c71701789"
uuid = "082447d4-558c-5d27-93f4-14fc19e9eca2"
version = "1.73.0"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra"]
git-tree-sha1 = "12177ad6b3cad7fd50c8b3825ce24a99ad61c18f"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.26.1"
weakdeps = ["SparseArrays"]

    [deps.ChainRulesCore.extensions]
    ChainRulesCoreSparseArraysExt = "SparseArrays"

[[deps.Clustering]]
deps = ["Distances", "LinearAlgebra", "NearestNeighbors", "Printf", "Random", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "3e22db924e2945282e70c33b75d4dde8bfa44c94"
uuid = "aaaa29a8-35af-508c-8bc3-b662a17a0fe5"
version = "0.15.8"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "962834c22b66e32aa10f7611c08c8ca4e20749a9"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.8"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "b0fd3f56fa442f81e0a47815c92245acfaaa4e34"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.31.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "b10d0b65641d57b8b4d5e234446582de5047050d"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.5"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "a1f44953f2382ebb937d60dafbe2deea4bd23249"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.10.0"
weakdeps = ["SpecialFunctions"]

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "37ea44092930b1811e666c3bc38065d7d87fcc74"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.13.1"

[[deps.Combinatorics]]
git-tree-sha1 = "08c8b6831dc00bfea825826be0bc8336fc369860"
uuid = "861a8166-3701-5b0c-9a16-15d98fcdc6aa"
version = "1.0.2"

[[deps.CommonSubexpressions]]
deps = ["MacroTools"]
git-tree-sha1 = "cda2cfaebb4be89c9084adaca7dd7333369715c5"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.1"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "9d8a54ce4b17aa5bdce0ea5c34bc5e7c340d16ad"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.18.1"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.3.0+1"

[[deps.CompositionsBase]]
git-tree-sha1 = "802bb88cd69dfd1509f6670416bd4434015693ad"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.2"
weakdeps = ["InverseFunctions"]

    [deps.CompositionsBase.extensions]
    CompositionsBaseInverseFunctionsExt = "InverseFunctions"

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "21d088c496ea22914fe80906eb5bce65755e5ec8"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.5.1"

[[deps.ConstructionBase]]
git-tree-sha1 = "b4b092499347b18a015186eae3042f72267106cb"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.6.0"

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseLinearAlgebraExt = "LinearAlgebra"
    ConstructionBaseStaticArraysExt = "StaticArrays"

    [deps.ConstructionBase.weakdeps]
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.ContextVariablesX]]
deps = ["Compat", "Logging", "UUIDs"]
git-tree-sha1 = "25cc3803f1030ab855e383129dcd3dc294e322cc"
uuid = "6add18c4-b38d-439d-96f6-d6bc489c04c5"
version = "0.1.3"

[[deps.Contour]]
git-tree-sha1 = "439e35b0b36e2e5881738abc8857bd92ad6ff9a8"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.3"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DSP]]
deps = ["Bessels", "FFTW", "IterTools", "LinearAlgebra", "Polynomials", "Random", "Reexport", "SpecialFunctions", "Statistics"]
git-tree-sha1 = "5989debfc3b38f736e69724818210c67ffee4352"
uuid = "717857b8-e6f2-59f4-9121-6e50c889abd2"
version = "0.8.4"

    [deps.DSP.extensions]
    OffsetArraysExt = "OffsetArrays"

    [deps.DSP.weakdeps]
    OffsetArrays = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "DataStructures", "Future", "InlineStrings", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrecompileTools", "PrettyTables", "Printf", "Random", "Reexport", "SentinelArrays", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "5fab31e2e01e70ad66e3e24c968c264d1cf166d6"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.8.2"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "4e1fe97fdaed23e9dc21d4d664bea76b65fc50a0"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.22"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.Dbus_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "473e9afc9cf30814eb67ffa5f2db7df82c3ad9fd"
uuid = "ee1fde0b-3d02-5ea6-8484-8dfef6360eab"
version = "1.16.2+0"

[[deps.DefineSingletons]]
git-tree-sha1 = "0fba8b706d0178b4dc7fd44a96a92382c9065c2c"
uuid = "244e2a9f-e319-4986-a169-4d1fe445cd52"
version = "0.1.2"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.DiffResults]]
deps = ["StaticArraysCore"]
git-tree-sha1 = "782dd5f4561f5d267313f23853baaaa4c52ea621"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.1.0"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "79a2aca180a85c690c58a020d47b426954b590f8"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.16.0"

[[deps.Distances]]
deps = ["LinearAlgebra", "Statistics", "StatsAPI"]
git-tree-sha1 = "c7e3a542b999843086e2f29dac96a618c105be1d"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.12"
weakdeps = ["ChainRulesCore", "SparseArrays"]

    [deps.Distances.extensions]
    DistancesChainRulesCoreExt = "ChainRulesCore"
    DistancesSparseArraysExt = "SparseArrays"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"
version = "1.11.0"

[[deps.Distributions]]
deps = ["FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "Test"]
git-tree-sha1 = "c2614fa3aafe03d1a44b8e16508d9be718b8095a"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.89"

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"

    [deps.Distributions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DensityInterface = "b429d917-457f-4dbc-8f4c-0cc954292b1d"

[[deps.DocStringExtensions]]
git-tree-sha1 = "7442a5dfe1ebb773c29cc2962a8980f47221d76c"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.5"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.7.0"

[[deps.EpollShim_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a4be429317c42cfae6a7fc03c31bad1970c310d"
uuid = "2702e6a9-849d-5ed8-8c21-79e8b8f9ee43"
version = "0.0.20230411+1"

[[deps.ExceptionUnwrapping]]
deps = ["Test"]
git-tree-sha1 = "d36f682e590a83d63d1c7dbd287573764682d12a"
uuid = "460bff9d-24e4-43bc-9d9f-a8973cb893f4"
version = "0.1.11"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c307cd83373868391f3ac30b41530bc5d5d05d08"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.8.1+0"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "95ecf07c2eea562b5adbd0696af6db62c0f52560"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.5"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libva_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "cac41ca6b2d399adfc95e51240566f8a60a80806"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "8.1.0+0"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "Libdl", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "97f08406df914023af55ade2f843c39e99c5d969"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.10.0"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6866aec60ef98e3164cd8d6855225684207e9dff"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.12+0"

[[deps.FLoops]]
deps = ["BangBang", "Compat", "FLoopsBase", "InitialValues", "JuliaVariables", "MLStyle", "Serialization", "Setfield", "Transducers"]
git-tree-sha1 = "0a2e5873e9a5f54abb06418d57a8df689336a660"
uuid = "cc61a311-1640-44b5-9fba-1b764f453329"
version = "0.2.2"

[[deps.FLoopsBase]]
deps = ["ContextVariablesX"]
git-tree-sha1 = "656f7a6859be8673bf1f35da5670246b923964f7"
uuid = "b9860ae5-e623-471e-878b-f6a53c775ea6"
version = "0.1.1"

[[deps.FilePathsBase]]
deps = ["Compat", "Dates"]
git-tree-sha1 = "3bab2c5aa25e7840a4b065805c0cdfc01f3068d2"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.24"
weakdeps = ["Mmap", "Test"]

    [deps.FilePathsBase.extensions]
    FilePathsBaseMmapExt = "Mmap"
    FilePathsBaseTestExt = "Test"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FillArrays]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "2f979084d1e13948a3352cf64a25df6bd3b4dca3"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.16.0"
weakdeps = ["PDMats", "SparseArrays", "StaticArrays", "Statistics"]

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStaticArraysExt = "StaticArrays"
    FillArraysStatisticsExt = "Statistics"

[[deps.FixedPointNumbers]]
deps = ["Random", "Statistics"]
git-tree-sha1 = "59af96b98217c6ef4ae0dfe065ac7c20831d1a84"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.6"

[[deps.Flux]]
deps = ["Adapt", "ChainRulesCore", "Compat", "Functors", "LinearAlgebra", "MLDataDevices", "MLUtils", "MacroTools", "NNlib", "OneHotArrays", "Optimisers", "Preferences", "ProgressLogging", "Random", "Reexport", "Setfield", "SparseArrays", "SpecialFunctions", "Statistics", "Zygote"]
git-tree-sha1 = "df520a0727f843576801a0294f5be1a94be28e23"
uuid = "587475ba-b771-5e3f-ad9e-33799f191a9c"
version = "0.14.25"

    [deps.Flux.extensions]
    FluxAMDGPUExt = "AMDGPU"
    FluxCUDAExt = "CUDA"
    FluxCUDAcuDNNExt = ["CUDA", "cuDNN"]
    FluxEnzymeExt = "Enzyme"
    FluxMPIExt = "MPI"
    FluxMPINCCLExt = ["CUDA", "MPI", "NCCL"]

    [deps.Flux.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
    MPI = "da04e1cc-30fd-572f-bb4f-1f8673147195"
    NCCL = "3fe64909-d7a1-4096-9b7d-7a0f12cf0f6b"
    cuDNN = "02a925ec-e4fe-4b08-9a7e-0d78e3d38ccd"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Zlib_jll"]
git-tree-sha1 = "f85dac9a96a01087df6e3a749840015a0ca3817d"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.17.1+0"

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions"]
git-tree-sha1 = "cddeab6487248a39dae1a960fff0ac17b2a28888"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "1.3.3"
weakdeps = ["StaticArrays"]

    [deps.ForwardDiff.extensions]
    ForwardDiffStaticArraysExt = "StaticArrays"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "70329abc09b886fd2c5d94ad2d9527639c421e3e"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.14.3+1"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7a214fdac5ed5f59a22c2d9a885a16da1c74bbc7"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.17+0"

[[deps.Functors]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "64d8e93700c7a3f28f717d265382d52fac9fa1c1"
uuid = "d9f16b24-f501-4c13-a1f2-28368ffc5196"
version = "0.4.12"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"
version = "1.11.0"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll", "libdecor_jll", "xkbcommon_jll"]
git-tree-sha1 = "9e0fb9e54594c47f278d75063980e43066e26e20"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.4.1+1"

[[deps.GPUArrays]]
deps = ["Adapt", "GPUArraysCore", "KernelAbstractions", "LLVM", "LinearAlgebra", "Printf", "Random", "Reexport", "ScopedValues", "Serialization", "Statistics"]
git-tree-sha1 = "be941842a40b6daac98496994ea69054ba4c5144"
uuid = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"
version = "11.2.3"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "83cf05ab16a73219e5f6bd1bdfa9848fa24ac627"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.2.0"

[[deps.GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Preferences", "Printf", "Qt6Wayland_jll", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "p7zip_jll"]
git-tree-sha1 = "f954322d5de03ec630d177cda203dcd92b6be399"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.73.26"

    [deps.GR.extensions]
    IJuliaExt = "IJulia"

    [deps.GR.weakdeps]
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "FreeType2_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt6Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "6fada551286ab6ea4ca1628cb2de9f166a2ec966"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.73.26+0"

[[deps.GettextRuntime_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll"]
git-tree-sha1 = "45288942190db7c5f760f59c04495064eedf9340"
uuid = "b0724c58-0f36-5564-988d-3bb0596ebc4a"
version = "0.22.4+0"

[[deps.Ghostscript_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Zlib_jll"]
git-tree-sha1 = "38044a04637976140074d0b0621c1edf0eb531fd"
uuid = "61579ee1-b43e-5ca0-a5da-69d92c66a64b"
version = "9.55.1+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "GettextRuntime_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "24f6def62397474a297bfcec22384101609142ed"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.86.3+0"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a6dbda1fd736d60cc477d99f2e7a042acfa46e8"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.15+0"

[[deps.Graphs]]
deps = ["ArnoldiMethod", "DataStructures", "Distributed", "Inflate", "LinearAlgebra", "Random", "SharedArrays", "SimpleTraits", "SparseArrays", "Statistics"]
git-tree-sha1 = "c5abfa0ae0aaee162a3fbb053c13ecda39be545b"
uuid = "86223c79-3864-5bf0-83f7-82e725a168b6"
version = "1.13.0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "ConcurrentUtilities", "Dates", "ExceptionUnwrapping", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "PrecompileTools", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "51059d23c8bb67911a2e6fd5130229113735fc7e"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.11.0"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "f923f9a774fcf3f5cb761bfa43aeadd689714813"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "8.5.1+0"

[[deps.HashArrayMappedTries]]
git-tree-sha1 = "2eaa69a7cab70a52b9687c8bf950a5a93ec895ae"
uuid = "076d061b-32b6-4027-95e0-9a2c6f6d7e74"
version = "0.2.0"

[[deps.HypergeometricFunctions]]
deps = ["LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "68c173f4f449de5b438ee67ed0c9c748dc31a2ec"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.28"

[[deps.IRTools]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "57e9ce6cf68d0abf5cb6b3b4abf9bedf05c939c0"
uuid = "7869d1d1-7146-5819-86e3-90919afe41df"
version = "0.4.15"

[[deps.Inflate]]
git-tree-sha1 = "d1b1b796e47d94588b3757fe84fbf65a5ec4a80d"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.5"

[[deps.InitialValues]]
git-tree-sha1 = "4da0f88e9a39111c2fa3add390ab15f3a44f3ca3"
uuid = "22cec73e-a1b8-11e9-2c92-598750a2cf9c"
version = "0.3.1"

[[deps.InlineStrings]]
git-tree-sha1 = "8f3d257792a522b4601c24a577954b0a8cd7334d"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.4.5"

    [deps.InlineStrings.extensions]
    ArrowTypesExt = "ArrowTypes"
    ParsersExt = "Parsers"

    [deps.InlineStrings.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"
    Parsers = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl"]
git-tree-sha1 = "ec1debd61c300961f98064cfb21287613ad7f303"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2025.2.0+0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.InverseFunctions]]
git-tree-sha1 = "a779299d77cd080bf77b97535acecd73e1c5e5cb"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.17"
weakdeps = ["Dates", "Test"]

    [deps.InverseFunctions.extensions]
    InverseFunctionsDatesExt = "Dates"
    InverseFunctionsTestExt = "Test"

[[deps.InvertedIndices]]
git-tree-sha1 = "6da3c4316095de0f5ee2ebd875df8721e7e0bdbe"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.3.1"

[[deps.IrrationalConstants]]
git-tree-sha1 = "b2d91fe939cae05960e760110b328288867b5758"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.6"

[[deps.IterTools]]
git-tree-sha1 = "42d5f897009e7ff2cf88db414a389e5ed1bdd023"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.10.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLFzf]]
deps = ["REPL", "Random", "fzf_jll"]
git-tree-sha1 = "82f7acdc599b65e0f8ccd270ffa1467c21cb647b"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.11"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "7204148362dafe5fe6a273f855b8ccbe4df8173e"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.8.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c0c9b76f3520863909825cbecdef58cd63de705a"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.1.5+0"

[[deps.JuliaSyntaxHighlighting]]
deps = ["StyledStrings"]
uuid = "ac6e5ff7-fb65-4e79-a425-ec3bc9c03011"
version = "1.12.0"

[[deps.JuliaVariables]]
deps = ["MLStyle", "NameResolution"]
git-tree-sha1 = "49fb3cb53362ddadb4415e9b73926d6b40709e70"
uuid = "b14d175d-62b4-44ba-8fb7-3064adc8c3ec"
version = "0.2.4"

[[deps.KernelAbstractions]]
deps = ["Adapt", "Atomix", "InteractiveUtils", "MacroTools", "PrecompileTools", "Requires", "StaticArrays", "UUIDs"]
git-tree-sha1 = "f2e76d3ced51a2a9e185abc0b97494c7273f649f"
uuid = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
version = "0.9.41"

    [deps.KernelAbstractions.extensions]
    EnzymeExt = "EnzymeCore"
    LinearAlgebraExt = "LinearAlgebra"
    SparseArraysExt = "SparseArrays"

    [deps.KernelAbstractions.weakdeps]
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "059aabebaa7c82ccb853dd4a0ee9d17796f7e1bc"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.3+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "17b94ecafcfa45e8360a4fc9ca6b583b049e4e37"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "4.1.0+0"

[[deps.LLVM]]
deps = ["CEnum", "LLVMExtra_jll", "Libdl", "Preferences", "Printf", "Unicode"]
git-tree-sha1 = "d422dfd9707bec6617335dc2ea3c5172a87d5908"
uuid = "929cbde3-209d-540e-8aea-75f648917ca0"
version = "9.1.3"

    [deps.LLVM.extensions]
    BFloat16sExt = "BFloat16s"

    [deps.LLVM.weakdeps]
    BFloat16s = "ab4f0b2a-ad5b-11e8-123f-65d77653426b"

[[deps.LLVMExtra_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "05a8bd5a42309a9ec82f700876903abce1017dd3"
uuid = "dad2f222-ce93-54a1-a47d-0025e8a3acab"
version = "0.0.34+0"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "eb62a3deb62fc6d8822c0c4bef73e4412419c5d8"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "18.1.8+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "dda21b8cbd6a6c40d9d02a73230f9d70fed6918c"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.0"

[[deps.Latexify]]
deps = ["Format", "Ghostscript_jll", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Requires"]
git-tree-sha1 = "44f93c47f9cd6c7e431f2f2091fcba8f01cd7e8f"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.10"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    SparseArraysExt = "SparseArrays"
    SymEngineExt = "SymEngine"
    TectonicExt = "tectonic_jll"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"
    tectonic_jll = "d7dd28d6-a5e6-559c-9131-7eb760cdacc5"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"
version = "1.11.0"

[[deps.LearnAPI]]
deps = ["InteractiveUtils", "Statistics"]
git-tree-sha1 = "ec695822c1faaaa64cee32d0b21505e1977b4809"
uuid = "92ad9a40-7767-427a-9ee6-6e577f1266cb"
version = "0.1.0"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "OpenSSL_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.15.0+0"

[[deps.LibGit2]]
deps = ["LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "OpenSSL_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.9.0+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "OpenSSL_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.3+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c8da7e6a91781c41a863611c7e966098d783c57a"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.4.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "d36c21b9e7c172a44a10484125024495e2625ac0"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.7.1+1"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "be484f5c92fad0bd8acfef35fe017900b0b73809"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.18.0+0"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "cc3ad4faf30015a3e8094c9b5b7f19e85bdf2386"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.42.0+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "f04133fe05eff1667d2054c53d59f9122383fe05"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.7.2+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "d620582b1f0cbe2c72dd1d5bd195a9ce73370ab1"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.42.0+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.12.0"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "13ca9e2586b89836fd20cccf56e57e2b9ae7f38f"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.29"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "f00544d95982ea270145636c181ceda21c4e2575"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.2.0"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "oneTBB_jll"]
git-tree-sha1 = "282cadc186e7b2ae0eeadbd7a4dffed4196ae2aa"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2025.2.0+0"

[[deps.MLCore]]
deps = ["DataAPI", "SimpleTraits", "Tables"]
git-tree-sha1 = "73907695f35bc7ffd9f11f6c4f2ee8c1302084be"
uuid = "c2834f40-e789-41da-a90e-33b280584a8c"
version = "1.0.0"

[[deps.MLDataDevices]]
deps = ["Adapt", "Compat", "Functors", "Preferences", "Random"]
git-tree-sha1 = "85b47bc5a8bf0c886286638585df3bec7c9f8269"
uuid = "7e8f7934-dd98-4c1a-8fe8-92b47a384d40"
version = "1.5.3"

    [deps.MLDataDevices.extensions]
    MLDataDevicesAMDGPUExt = "AMDGPU"
    MLDataDevicesCUDAExt = "CUDA"
    MLDataDevicesChainRulesCoreExt = "ChainRulesCore"
    MLDataDevicesChainRulesExt = "ChainRules"
    MLDataDevicesFillArraysExt = "FillArrays"
    MLDataDevicesGPUArraysExt = "GPUArrays"
    MLDataDevicesMLUtilsExt = "MLUtils"
    MLDataDevicesMetalExt = ["GPUArrays", "Metal"]
    MLDataDevicesOneHotArraysExt = "OneHotArrays"
    MLDataDevicesReactantExt = "Reactant"
    MLDataDevicesRecursiveArrayToolsExt = "RecursiveArrayTools"
    MLDataDevicesReverseDiffExt = "ReverseDiff"
    MLDataDevicesSparseArraysExt = "SparseArrays"
    MLDataDevicesTrackerExt = "Tracker"
    MLDataDevicesZygoteExt = "Zygote"
    MLDataDevicescuDNNExt = ["CUDA", "cuDNN"]
    MLDataDevicesoneAPIExt = ["GPUArrays", "oneAPI"]

    [deps.MLDataDevices.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    ChainRules = "082447d4-558c-5d27-93f4-14fc19e9eca2"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    FillArrays = "1a297f60-69ca-5386-bcde-b61e274b549b"
    GPUArrays = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"
    MLUtils = "f1d291b0-491e-4a28-83b9-f70985020b54"
    Metal = "dde4c033-4e86-420c-a63e-0dd931031962"
    OneHotArrays = "0b1bfda6-eb8a-41d2-88d8-f5af5cad476f"
    Reactant = "3c362404-f566-11ee-1572-e11a4b42c853"
    RecursiveArrayTools = "731186ca-8d62-57ce-b412-fbd966d074cd"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"
    cuDNN = "02a925ec-e4fe-4b08-9a7e-0d78e3d38ccd"
    oneAPI = "8f75cd03-7ff8-4ecb-9b8f-daf728133b1b"

[[deps.MLStyle]]
git-tree-sha1 = "bc38dff0548128765760c79eb7388a4b37fae2c8"
uuid = "d8e11817-5142-5d16-987a-aa16d5891078"
version = "0.4.17"

[[deps.MLUtils]]
deps = ["ChainRulesCore", "Compat", "DataAPI", "DelimitedFiles", "FLoops", "MLCore", "NNlib", "Random", "ShowCases", "SimpleTraits", "Statistics", "StatsBase", "Tables", "Transducers"]
git-tree-sha1 = "a772d8d1987433538a5c226f79393324b55f7846"
uuid = "f1d291b0-491e-4a28-83b9-f70985020b54"
version = "0.4.8"

[[deps.MacroTools]]
git-tree-sha1 = "1e0228a030642014fe5cfe68c2c0a818f9e3f522"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.16"

[[deps.ManifoldLearning]]
deps = ["Combinatorics", "Graphs", "LinearAlgebra", "MultivariateStats", "Random", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "4c5564c707899c3b6bc6d324b05e43eb7f277f2b"
uuid = "06eb3307-b2af-5a2a-abea-d33192699d32"
version = "0.9.0"

[[deps.Markdown]]
deps = ["Base64", "JuliaSyntaxHighlighting", "StyledStrings"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "NetworkOptions", "Random", "Sockets"]
git-tree-sha1 = "8785729fa736197687541f7053f6d8ab7fc44f92"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.10"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "ff69a2b1330bcb730b9ac1ab7dd680176f5896b8"
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.1010+0"

[[deps.Measures]]
git-tree-sha1 = "b513cedd20d9c914783d8ad83d08120702bf2c77"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.3"

[[deps.MicroCollections]]
deps = ["Accessors", "BangBang", "InitialValues"]
git-tree-sha1 = "44d32db644e84c75dab479f1bc15ee76a1a3618f"
uuid = "128add7d-3638-4c79-886c-908ea0c25c34"
version = "0.2.0"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2025.11.4"

[[deps.MultivariateStats]]
deps = ["Arpack", "LinearAlgebra", "SparseArrays", "Statistics", "StatsAPI", "StatsBase"]
git-tree-sha1 = "7008a3412d823e29d370ddc77411d593bd8a3d03"
uuid = "6f286f6a-111f-5878-ab1e-185364afe411"
version = "0.9.1"

[[deps.NNlib]]
deps = ["Adapt", "Atomix", "ChainRulesCore", "GPUArraysCore", "KernelAbstractions", "LinearAlgebra", "Random", "ScopedValues", "Statistics"]
git-tree-sha1 = "78cd28dbd5f03f99ccaba45c987107adcb61c115"
uuid = "872c559c-99b0-510c-b3b7-b6c96a88d5cd"
version = "0.9.34"

    [deps.NNlib.extensions]
    NNlibAMDGPUExt = "AMDGPU"
    NNlibCUDACUDNNExt = ["CUDA", "cuDNN"]
    NNlibCUDAExt = "CUDA"
    NNlibEnzymeCoreExt = "EnzymeCore"
    NNlibFFTWExt = "FFTW"
    NNlibForwardDiffExt = "ForwardDiff"
    NNlibMetalExt = "Metal"
    NNlibSpecialFunctionsExt = "SpecialFunctions"

    [deps.NNlib.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    Metal = "dde4c033-4e86-420c-a63e-0dd931031962"
    SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"
    cuDNN = "02a925ec-e4fe-4b08-9a7e-0d78e3d38ccd"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "dbd2e8cd2c1c27f0b584f6661b4309609c5a685e"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.1.4"

[[deps.NameResolution]]
deps = ["PrettyPrint"]
git-tree-sha1 = "1a0fa0e9613f46c9b8c11eee38ebb4f590013c5e"
uuid = "71a1bf82-56d0-4bbc-8a3c-48b961074391"
version = "0.1.5"

[[deps.NearestNeighbors]]
deps = ["Distances", "StaticArrays"]
git-tree-sha1 = "ca7e18198a166a1f3eb92a3650d53d94ed8ca8a1"
uuid = "b8a86587-4115-5ab1-83bc-aa920d37bbce"
version = "0.4.22"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.3.0"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b6aa4566bb7ae78498a5e68943863fa8b5231b59"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.6+0"

[[deps.OneHotArrays]]
deps = ["Adapt", "ChainRulesCore", "Compat", "GPUArraysCore", "LinearAlgebra", "NNlib"]
git-tree-sha1 = "a0872718ad79c8fe282bd06374d88b57577330fd"
uuid = "0b1bfda6-eb8a-41d2-88d8-f5af5cad476f"
version = "0.2.8"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.29+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.7+0"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "NetworkOptions", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "1d1aaa7d449b58415f97d2839c318b70ffb525a0"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.6.1"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.5.4+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1346c9208249809840c91b26703912dff463d335"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.6+0"

[[deps.Optimisers]]
deps = ["ChainRulesCore", "Functors", "LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "c9ff5c686240c31eb8570b662dd1f66f4b183116"
uuid = "3bd65402-5787-11e9-1adc-39752487f4e2"
version = "0.3.4"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e2bb57a313a74b8104064b7efd01406c0a50d2ff"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.6.1+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "94ba93778373a53bfd5a0caaf7d809c445292ff4"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.8.2"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.44.0+1"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "f07c06228a1c670ae4c87d1276b92c7c597fdda0"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.35"

[[deps.Pango_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "FriBidi_jll", "Glib_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "58e5ed5e386e156bd93e86b305ebd21ac63d2d04"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.57.1+0"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "468dbe2b510c876dc091b2c74ed52c7c34f48b9b"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.5"

[[deps.Peaks]]
deps = ["RecipesBase", "SIMD"]
git-tree-sha1 = "75d0ce1c30696d77bc60840222d7fc5d549ebf5f"
uuid = "18e31ff7-3703-566c-8e60-38913d67486b"
version = "0.5.3"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "e4a6721aa89e62e5d4217c0b21bd714263779dda"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.46.4+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.12.1"
weakdeps = ["REPL"]

    [deps.Pkg.extensions]
    REPLExt = "REPL"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "41031ef3a1be6f5bbbf3e8073f210556daeae5ca"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.3.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "StableRNGs", "Statistics"]
git-tree-sha1 = "26ca162858917496748aad52bb5d3be4d26a228a"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.4.4"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "PrecompileTools", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "TOML", "UUIDs", "UnicodeFun", "Unzip"]
git-tree-sha1 = "cb20a4eacda080e517e4deb9cfb6c7c518131265"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.41.6"

    [deps.Plots.extensions]
    FileIOExt = "FileIO"
    GeometryBasicsExt = "GeometryBasics"
    IJuliaExt = "IJulia"
    ImageInTerminalExt = "ImageInTerminal"
    UnitfulExt = "Unitful"

    [deps.Plots.weakdeps]
    FileIO = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
    GeometryBasics = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"
    ImageInTerminal = "d8c32880-2388-543b-8c61-d9f865259254"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.Polynomials]]
deps = ["LinearAlgebra", "OrderedCollections", "Setfield", "SparseArrays"]
git-tree-sha1 = "2d99b4c8a7845ab1342921733fa29366dae28b24"
uuid = "f27b6e38-b328-58d1-80ce-0feddd5e7a45"
version = "4.1.1"

    [deps.Polynomials.extensions]
    PolynomialsChainRulesCoreExt = "ChainRulesCore"
    PolynomialsFFTWExt = "FFTW"
    PolynomialsMakieExt = "Makie"
    PolynomialsMutableArithmeticsExt = "MutableArithmetics"
    PolynomialsRecipesBaseExt = "RecipesBase"

    [deps.Polynomials.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
    Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
    MutableArithmetics = "d8a4904e-b15c-11e9-3269-09a3773c0cb0"
    RecipesBase = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "36d8b4b899628fb92c2749eb488d884a926614d3"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.3"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "edbeefc7a4889f528644251bdb5fc9ab5348bc2c"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.3.4"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "8b770b60760d4451834fe79dd483e318eee709c4"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.5.2"

[[deps.PrettyPrint]]
git-tree-sha1 = "632eb4abab3449ab30c5e1afaa874f0b98b586e4"
uuid = "8162dcfd-2161-5ef2-ae6c-7681170c5f98"
version = "0.2.0"

[[deps.PrettyTables]]
deps = ["Crayons", "LaTeXStrings", "Markdown", "PrecompileTools", "Printf", "Reexport", "StringManipulation", "Tables"]
git-tree-sha1 = "1101cd475833706e4d0e7b122218257178f48f34"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "2.4.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.ProgressLogging]]
deps = ["Logging", "SHA", "UUIDs"]
git-tree-sha1 = "80d919dee55b9c50e8d9e2da5eeafff3fe58b539"
uuid = "33c8b6b6-d38a-422a-b730-caa89a2f386c"
version = "0.1.4"

[[deps.Qt6Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Vulkan_Loader_jll", "Xorg_libSM_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_cursor_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "libinput_jll", "xkbcommon_jll"]
git-tree-sha1 = "144895f6166994730ee7ff8113b981fc360638f1"
uuid = "c0090381-4147-56d7-9ebc-da0b1113ec56"
version = "6.10.2+2"

[[deps.Qt6Declarative_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6ShaderTools_jll", "Qt6Svg_jll"]
git-tree-sha1 = "d5b7dd0e226774cbd87e2790e34def09245c7eab"
uuid = "629bc702-f1f5-5709-abd5-49b8460ea067"
version = "6.10.2+1"

[[deps.Qt6ShaderTools_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll"]
git-tree-sha1 = "4d85eedf69d875982c46643f6b4f66919d7e157b"
uuid = "ce943373-25bb-56aa-8eca-768745ed7b5a"
version = "6.10.2+1"

[[deps.Qt6Svg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll"]
git-tree-sha1 = "81587ff5ff25a4e1115ce191e36285ede0334c9d"
uuid = "6de9746b-f93d-5813-b365-ba18ad4a9cf3"
version = "6.10.2+0"

[[deps.Qt6Wayland_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6Declarative_jll"]
git-tree-sha1 = "672c938b4b4e3e0169a07a5f227029d4905456f2"
uuid = "e99dba38-086e-5de3-a5b1-6e4c66e897c3"
version = "6.10.2+1"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "5e8e8b0ab68215d7a2b14b9921a946fee794749e"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.11.3"

    [deps.QuadGK.extensions]
    QuadGKEnzymeExt = "Enzyme"

    [deps.QuadGK.weakdeps]
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"

[[deps.REPL]]
deps = ["InteractiveUtils", "JuliaSyntaxHighlighting", "Markdown", "Sockets", "StyledStrings", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.RealDot]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "9f0a1b71baaf7650f4fa8a1d168c7fb6ee41f0c9"
uuid = "c1ae055f-0cd5-4b69-90a6-9a35b1a98df9"
version = "0.1.0"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "PrecompileTools", "RecipesBase"]
git-tree-sha1 = "45cf9fd0ca5839d06ef333c8201714e888486342"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.6.12"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "ffdaf70d81cf6ff22c2b6e733c900c3321cab864"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.1"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "62389eeff14780bfe55195b7204c0d8738436d64"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.1"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "5b3d50eb374cea306873b371d3f8d3915a018f0b"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.9.0"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "58cdd8fb2201a6267e1db87ff148dd6c1dbd8ad8"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.5.1+0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SIMD]]
deps = ["PrecompileTools"]
git-tree-sha1 = "e24dc23107d426a096d3eae6c165b921e74c18e4"
uuid = "fdea26ae-647d-5447-a871-4b548cad5224"
version = "3.7.2"

[[deps.ScientificTypes]]
deps = ["CategoricalArrays", "ColorTypes", "Dates", "Distributions", "PrettyTables", "Reexport", "ScientificTypesBase", "StatisticalTraits", "Tables"]
git-tree-sha1 = "75ccd10ca65b939dab03b812994e571bf1e3e1da"
uuid = "321657f4-b219-11e9-178b-2701a2544e81"
version = "3.0.2"

[[deps.ScientificTypesBase]]
git-tree-sha1 = "a8e18eb383b5ecf1b5e6fc237eb39255044fd92b"
uuid = "30f210dd-8aff-4c5f-94ba-8e64358c1161"
version = "3.0.0"

[[deps.ScopedValues]]
deps = ["HashArrayMappedTries", "Logging"]
git-tree-sha1 = "67a144433c4ce877ee6d1ada69a124d6b1ecf7be"
uuid = "7e506255-f358-4e82-b7e4-beb19740aa63"
version = "1.6.2"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "9b81b8393e50b7d4e6d0a9f14e192294d3b7c109"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.3.0"

[[deps.SegyIO]]
deps = ["Distributed", "Printf", "Test"]
git-tree-sha1 = "0fc24db28695a80aa59c179372b11165d371188a"
uuid = "157a0f19-4d44-4de5-a0d0-07e2f0ac4dfa"
version = "0.8.5"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "084c47c7c5ce5cfecefa0a98dff69eb3646b5a80"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.4.10"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "c5391c6ace3bc430ca630251d02ea9687169ca68"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.2"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"
version = "1.11.0"

[[deps.ShowCases]]
git-tree-sha1 = "7f534ad62ab2bd48591bdeac81994ea8c445e4a5"
uuid = "605ecd9f-84a6-4c9e-81e2-4798472b76a3"
version = "0.1.0"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SimpleBufferStream]]
git-tree-sha1 = "f305871d2f381d21527c770d4788c06c097c9bc1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.2.0"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "7ddb0b49c109481b046972c0e4ab02b2127d6a75"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.6"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "64d974c2e6fdf07f8155b5b2ca2ffa9069b608d9"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.2"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.12.0"

[[deps.SparseInverseSubset]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "52962839426b75b3021296f7df242e40ecfc0852"
uuid = "dc90abb0-5640-4711-901d-7e5b23a2fada"
version = "0.1.2"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "6547cbdd8ce32efba0d21c5a40fa96d1a3548f9f"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.8.0"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.SplittablesBase]]
deps = ["Setfield", "Test"]
git-tree-sha1 = "e08a62abc517eb79667d0a29dc08a3b589516bb5"
uuid = "171d559e-b47b-412a-8079-5efa626c420e"
version = "0.1.15"

[[deps.StableRNGs]]
deps = ["Random"]
git-tree-sha1 = "4f96c596b8c8258cc7d3b19797854d368f243ddc"
uuid = "860ef19b-820b-49d6-a774-d7a799459cd3"
version = "1.0.4"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "246a8bb2e6667f832eea063c3a56aef96429a3db"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.18"
weakdeps = ["ChainRulesCore", "Statistics"]

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

[[deps.StaticArraysCore]]
git-tree-sha1 = "6ab403037779dae8c514bad259f32a447262455a"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.4"

[[deps.StatisticalMeasures]]
deps = ["CategoricalArrays", "CategoricalDistributions", "Distributions", "LearnAPI", "LinearAlgebra", "MacroTools", "OrderedCollections", "PrecompileTools", "ScientificTypesBase", "StatisticalMeasuresBase", "Statistics", "StatsBase"]
git-tree-sha1 = "c1d4318fa41056b839dfbb3ee841f011fa6e8518"
uuid = "a19d573c-0a75-4610-95b3-7071388c7541"
version = "0.1.7"

    [deps.StatisticalMeasures.extensions]
    LossFunctionsExt = "LossFunctions"
    ScientificTypesExt = "ScientificTypes"

    [deps.StatisticalMeasures.weakdeps]
    LossFunctions = "30fc2ffe-d236-52d8-8643-a9d8f7c094a7"
    ScientificTypes = "321657f4-b219-11e9-178b-2701a2544e81"

[[deps.StatisticalMeasuresBase]]
deps = ["CategoricalArrays", "InteractiveUtils", "MLUtils", "MacroTools", "OrderedCollections", "PrecompileTools", "ScientificTypesBase", "Statistics"]
git-tree-sha1 = "17dfb22e2e4ccc9cd59b487dce52883e0151b4d3"
uuid = "c062fc1d-0d66-479b-b6ac-8b44719de4cc"
version = "0.1.1"

[[deps.StatisticalTraits]]
deps = ["ScientificTypesBase"]
git-tree-sha1 = "542d979f6e756f13f862aa00b224f04f9e445f11"
uuid = "64bff920-2084-43da-a3e6-9bb72801c0c9"
version = "3.4.0"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"
weakdeps = ["SparseArrays"]

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "8d7530a38dbd2c397be7ddd01a424e4f411dcc41"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.2.2"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "d1bf48bfcc554a3761a133fe3a9bb01488e06916"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.21"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "91f091a8716a6bb38417a6e6f274602a19aaa685"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.5.2"
weakdeps = ["ChainRulesCore", "InverseFunctions"]

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

[[deps.StringManipulation]]
deps = ["PrecompileTools"]
git-tree-sha1 = "d05693d339e37d6ab134c5ab53c29fce5ee5d7d5"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.4.4"

[[deps.StructArrays]]
deps = ["ConstructionBase", "DataAPI", "Tables"]
git-tree-sha1 = "ad8002667372439f2e3611cfd14097e03fa4bccd"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.7.3"
weakdeps = ["Adapt", "GPUArraysCore", "KernelAbstractions", "LinearAlgebra", "SparseArrays", "StaticArrays"]

    [deps.StructArrays.extensions]
    StructArraysAdaptExt = "Adapt"
    StructArraysGPUArraysCoreExt = ["GPUArraysCore", "KernelAbstractions"]
    StructArraysLinearAlgebraExt = "LinearAlgebra"
    StructArraysSparseArraysExt = "SparseArrays"
    StructArraysStaticArraysExt = "StaticArrays"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.8.3+2"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "f2c1efbc8f3a609aadf318094f8fc5204bdaf344"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.12.1"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.TranscodingStreams]]
git-tree-sha1 = "0c45878dcfdcfa8480052b6ab162cdd138781742"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.11.3"

[[deps.Transducers]]
deps = ["Accessors", "ArgCheck", "BangBang", "Baselet", "CompositionsBase", "ConstructionBase", "DefineSingletons", "Distributed", "InitialValues", "Logging", "Markdown", "MicroCollections", "Requires", "SplittablesBase", "Tables"]
git-tree-sha1 = "7deeab4ff96b85c5f72c824cae53a1398da3d1cb"
uuid = "28d57a85-8fef-5791-bfe6-a80928e7c999"
version = "0.4.84"

    [deps.Transducers.extensions]
    TransducersAdaptExt = "Adapt"
    TransducersBlockArraysExt = "BlockArrays"
    TransducersDataFramesExt = "DataFrames"
    TransducersLazyArraysExt = "LazyArrays"
    TransducersOnlineStatsBaseExt = "OnlineStatsBase"
    TransducersReferenceablesExt = "Referenceables"

    [deps.Transducers.weakdeps]
    Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
    BlockArrays = "8e7c35d0-a365-5155-bbbb-fb81a777f24e"
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    LazyArrays = "5078a376-72f3-5289-bfd5-ec5146d43c02"
    OnlineStatsBase = "925886fa-5bf2-5e8e-b522-a9147a512338"
    Referenceables = "42d2dcc6-99eb-4e98-b66c-637b7d73030e"

[[deps.URIs]]
git-tree-sha1 = "bef26fb046d031353ef97a82e3fdb6afe7f21b1a"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.6.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.UnsafeAtomics]]
git-tree-sha1 = "0f30765c32d66d58e41f4cb5624d4fc8a82ec13b"
uuid = "013be700-e6cd-48c3-b4a1-df204f14c38f"
version = "0.3.1"
weakdeps = ["LLVM"]

    [deps.UnsafeAtomics.extensions]
    UnsafeAtomicsLLVM = ["LLVM"]

[[deps.Unzip]]
git-tree-sha1 = "ca0969166a028236229f63514992fc073799bb78"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.2.0"

[[deps.Vulkan_Loader_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Wayland_jll", "Xorg_libX11_jll", "Xorg_libXrandr_jll", "xkbcommon_jll"]
git-tree-sha1 = "2f0486047a07670caad3a81a075d2e518acc5c59"
uuid = "a44049a8-05dd-5a78-86c9-5fde0876e88c"
version = "1.3.243+0"

[[deps.Wayland_jll]]
deps = ["Artifacts", "EpollShim_jll", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "96478df35bbc2f3e1e791bc7a3d0eeee559e60e9"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.24.0+0"

[[deps.WeakRefStrings]]
deps = ["DataAPI", "InlineStrings", "Parsers"]
git-tree-sha1 = "0716e01c3b40413de5dedbc9c5c69f27cddfddfc"
uuid = "ea10d353-3f73-51f8-a26c-33c1cb351aa5"
version = "1.4.3"

[[deps.WorkerUtilities]]
git-tree-sha1 = "cd1659ba0d57b71a464a29e64dbc67cfe83d54e7"
uuid = "76eceee3-57b5-4d4a-8e66-0e911cebbf60"
version = "1.6.1"

[[deps.XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b29c22e245d092b8b4e8d3c09ad7baa586d9f573"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.8.3+0"

[[deps.Xorg_libICE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a3ea76ee3f4facd7a64684f9af25310825ee3668"
uuid = "f67eecfb-183a-506d-b269-f58e52b52d7c"
version = "1.1.2+0"

[[deps.Xorg_libSM_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libICE_jll"]
git-tree-sha1 = "9c7ad99c629a44f81e7799eb05ec2746abb5d588"
uuid = "c834827a-8449-5923-a945-d239c165b7dd"
version = "1.2.6+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "808090ede1d41644447dd5cbafced4731c56bd2f"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.13+0"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "aa1261ebbac3ccc8d16558ae6799524c450ed16b"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.13+0"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "6c74ca84bbabc18c4547014765d194ff0b4dc9da"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.4+0"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "52858d64353db33a56e13c341d7bf44cd0d7b309"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.6+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "1a4a26870bf1e5d26cd585e38038d399d7e65706"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.8+0"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "75e00946e43621e09d431d9b95818ee751e6b2ef"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "6.0.2+0"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "a376af5c7ae60d29825164db40787f15c80c7c54"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.8.3+0"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll"]
git-tree-sha1 = "0ba01bc7396896a4ace8aab67db31403c71628f4"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.7+0"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "6c174ef70c96c76f4c3f4d3cfbe09d018bcd1b53"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.6+0"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "7ed9347888fac59a618302ee38216dd0379c480d"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.12+0"

[[deps.Xorg_libpciaccess_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "58972370b81423fc546c56a60ed1a009450177c3"
uuid = "a65dc6b1-eb27-53a1-bb3e-dea574b5389e"
version = "0.19.0+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXau_jll", "Xorg_libXdmcp_jll"]
git-tree-sha1 = "bfcaf7ec088eaba362093393fe11aa141fa15422"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.17.1+0"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "ed756a03e95fff88d8f738ebc2849431bdd4fd1a"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.2.0+0"

[[deps.Xorg_xcb_util_cursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_jll", "Xorg_xcb_util_renderutil_jll"]
git-tree-sha1 = "9750dc53819eba4e9a20be42349a6d3b86c7cdf8"
uuid = "e920d4aa-a673-5f3a-b3d7-f755a4d47c43"
version = "0.1.6+0"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "f4fc02e384b74418679983a97385644b67e1263b"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.1+0"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll"]
git-tree-sha1 = "68da27247e7d8d8dafd1fcf0c3654ad6506f5f97"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.1+0"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "44ec54b0e2acd408b0fb361e1e9244c60c9c3dd4"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.1+0"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "5b0263b6d080716a02544c55fdff2c8d7f9a16a0"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.10+0"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "f233c83cad1fa0e70b7771e0e21b061a116f2763"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.2+0"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "801a858fc9fb90c11ffddee1801bb06a738bda9b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.7+0"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "ed349d26affcacafbc7fc2941ace1fb98f71e715"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.47.0+1"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a63799ff68005991f9d9491b6e95bd3478d783cb"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.6.0+0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.3.1+2"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "446b23e73536f84e8037f5dce465e92275f6a308"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.7+1"

[[deps.Zygote]]
deps = ["AbstractFFTs", "ChainRules", "ChainRulesCore", "DiffRules", "Distributed", "FillArrays", "ForwardDiff", "GPUArrays", "GPUArraysCore", "IRTools", "InteractiveUtils", "LinearAlgebra", "LogExpFunctions", "MacroTools", "NaNMath", "PrecompileTools", "Random", "Requires", "SparseArrays", "SpecialFunctions", "Statistics", "ZygoteRules"]
git-tree-sha1 = "3c73ed65f928f8602e9a30e93125b209133498a9"
uuid = "e88e6eb3-aa80-5325-afca-941959d7151f"
version = "0.6.76"

    [deps.Zygote.extensions]
    ZygoteColorsExt = "Colors"
    ZygoteDistancesExt = "Distances"
    ZygoteTrackerExt = "Tracker"

    [deps.Zygote.weakdeps]
    Colors = "5ae59095-9a9b-59fe-a467-6f913c188581"
    Distances = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.ZygoteRules]]
deps = ["ChainRulesCore", "MacroTools"]
git-tree-sha1 = "434b3de333c75fc446aa0d19fc394edafd07ab08"
uuid = "700de1a5-db45-46bc-99cf-38207098b444"
version = "0.2.7"

[[deps.eudev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c3b0e6196d50eab0c5ed34021aaa0bb463489510"
uuid = "35ca27e7-8b34-5b7f-bca9-bdc33f59eb06"
version = "3.2.14+0"

[[deps.fzf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b6a34e0e0960190ac2a4363a1bd003504772d631"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.61.1+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "850b06095ee71f0135d644ffd8a52850699581ed"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.13.3+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "125eedcb0a4a0bba65b657251ce1d27c8714e9d6"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.17.4+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.15.0+0"

[[deps.libdecor_jll]]
deps = ["Artifacts", "Dbus_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pango_jll", "Wayland_jll", "xkbcommon_jll"]
git-tree-sha1 = "9bf7903af251d2050b467f76bdbe57ce541f7f4f"
uuid = "1183f4f0-6f2a-5f1a-908b-139f9cdfea6f"
version = "0.2.2+0"

[[deps.libdrm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libpciaccess_jll"]
git-tree-sha1 = "63aac0bcb0b582e11bad965cef4a689905456c03"
uuid = "8e53e030-5e6c-5a89-a30b-be5b7263a166"
version = "2.4.125+1"

[[deps.libevdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "56d643b57b188d30cccc25e331d416d3d358e557"
uuid = "2db6ffa8-e38f-5e21-84af-90c45d0032cc"
version = "1.13.4+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "646634dd19587a56ee2f1199563ec056c5f228df"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.4+0"

[[deps.libinput_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "eudev_jll", "libevdev_jll", "mtdev_jll"]
git-tree-sha1 = "91d05d7f4a9f67205bd6cf395e488009fe85b499"
uuid = "36db933b-70db-51c0-b978-0f229ee0e533"
version = "1.28.1+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "e51150d5ab85cee6fc36726850f0e627ad2e4aba"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.58+0"

[[deps.libva_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll", "Xorg_libXext_jll", "Xorg_libXfixes_jll", "libdrm_jll"]
git-tree-sha1 = "7dbf96baae3310fe2fa0df0ccbb3c6288d5816c9"
uuid = "9a156e7d-b971-5f62-b2c9-67348b8fb97c"
version = "2.23.0+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll"]
git-tree-sha1 = "11e1772e7f3cc987e9d3de991dd4f6b2602663a5"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.8+0"

[[deps.mtdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b4d631fd51f2e9cdd93724ae25b2efc198b059b1"
uuid = "009596ad-96f7-51b1-9f1b-5ce2d5e8a71e"
version = "1.1.7+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.64.0+1"

[[deps.oneTBB_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl"]
git-tree-sha1 = "da8c1f6eee04831f14edcfa5dae611d309807e57"
uuid = "1317d2d5-d96f-522e-a858-c73665f53c3e"
version = "2022.3.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.7.0+0"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "14cc7083fc6dff3cc44f2bc435ee96d06ed79aa7"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "10164.0.1+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e7b67590c14d487e734dcb925924c5dc43ec85f3"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "4.1.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "a1fc6507a40bf504527d0d4067d718f8e179b2b8"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.13.0+0"
"""

# ╔═╡ Cell order:
# ╟─c76e8372-a529-11ef-03ba-a7de23d21592
# ╟─b3853692-e4c4-40aa-9c11-647d57de12e0
# ╟─29931cf2-a950-4c7b-b2c6-878689d6e5f8
# ╟─84e44b16-e046-4dd9-98e6-ff5863369d87
# ╟─14f8891d-fe53-4e5a-93fb-0c80b29631b5
# ╟─d6468e7e-2d5a-4973-b1e0-b2a5020966bf
# ╠═9488e212-9589-47c5-8999-9382a36d3548
# ╟─244906fc-ade7-41c1-aa39-260fa541073d
# ╠═f32c7867-492a-4404-8f3d-776d3eac3261
# ╠═6e2db287-84a2-4ecb-bc58-55a975b2cd1f
# ╠═770a98ad-782d-4998-8cb0-cc9831014869
# ╠═48c069f9-24ae-4dc3-8aa5-ff460ca3be81
# ╠═670e6291-f5a8-41ea-b342-2c270110909a
# ╠═ec324d61-bd97-4ac6-beb9-7c6e1aaa2213
# ╟─c423f3e0-846c-4148-a45f-551654a6b471
# ╠═499ba1e1-ec34-4ef7-ab9e-bf1c12a489c1
# ╠═59c37877-f77d-40eb-b6e6-9584fc76f500
# ╠═be428b20-0d19-4e17-be89-f69029c10285
# ╟─1c32cd0c-30be-4cbb-95ce-1676aea650f6
# ╠═83082979-d0d1-47aa-996b-bc1b8408f9fd
# ╠═858b9411-356e-443b-8a54-238a5899dd20
# ╠═f04e3f8d-5e5b-4f69-886e-2307fa4bb7e5
# ╠═686cf9b0-0dd3-4529-8d8e-86637857ba7c
# ╠═f81462e4-a76a-4f2c-9389-bb9b588c978d
# ╠═a88282a1-e3d4-4fa6-8080-ec72e45d6030
# ╠═dcbf426d-efb8-47c7-ba54-9759a17959cc
# ╠═00e25c16-8165-497e-95a9-4e81fed93e74
# ╠═11f00f03-0278-4e4e-ae7c-7ef65c81d3b9
# ╠═e3a06b53-fd6d-45fa-8e34-15e6bf29fdca
# ╠═6875bc49-bfd5-491f-b4c0-7fd491e18c2a
# ╠═ae2b7102-7c30-49e7-897b-0537085f24dd
# ╠═c8375cfe-5f90-49af-9ebb-6eadd70ac655
# ╟─c4b1290d-ca33-44ec-809d-2191a08ff206
# ╠═cd7d2586-aa65-4aff-bc5b-f5ab5b5663d8
# ╠═72f2eda2-5d34-4e32-9659-0d8b50bdd126
# ╟─3ffb9f00-f919-4d21-bfc0-3132538e9920
# ╟─2b1f908e-bf92-4b04-888d-f82539355298
# ╠═d9248fea-e85b-43a4-8ac3-fd2b3cbfab99
# ╠═cf999cbe-0711-41c4-99a0-d4cbb9c5cd8f
# ╠═e391d5a2-825b-4b74-856d-13294a7dc14f
# ╟─a288fc10-bcd8-4a7a-8aa4-f07f876e6bea
# ╠═fb34a3e6-f84b-4454-8cb9-6d625bdbe643
# ╠═bd10b539-dc98-42ca-aecd-df583377a3f0
# ╠═6616912c-576d-4ad2-b754-ea4a3e65ecb6
# ╠═b113f04e-67c2-41a8-8857-2cc466415f44
# ╠═4c1c5a5c-cf4e-4b59-8e5c-fb3a9a71b17e
# ╠═ba99833b-947a-4ca4-8142-ea928f3f5772
# ╠═21b63603-d840-48ea-a4da-4904e5f37678
# ╟─06b85900-2034-44fc-856c-da440e8b87a7
# ╠═e6339f81-3bd8-42d9-bfb4-4565fd44a6a0
# ╠═c02c1652-57d6-4089-8f34-a37e7072049c
# ╟─5b17ee99-c946-408c-a94a-41fb63a74f51
# ╟─49d698d5-d37c-4e9b-8fc1-402a263fb69c
# ╟─2ef52597-7ecd-4946-9bc6-505f989faa55
# ╠═1a750016-ff4e-4575-8764-39568c91a5db
# ╟─a9a597fa-9176-45e0-b494-bc00a0005c98
# ╠═2160f090-5040-47fa-9ae4-9d2fb6391c96
# ╟─200f314f-dae0-43be-82fa-b609a061142b
# ╠═6a9be9d5-1f4f-4e58-8c4d-5528cfc11cf4
# ╠═a9291696-b531-499e-8c96-c070e937fcf0
# ╠═a5f54d85-85e6-4060-970b-b8c93fececfb
# ╠═d34b12b6-e777-4311-9cc6-3f83124e57b9
# ╠═7d1954f6-4954-4e48-a934-9e87d0d29363
# ╠═65c6c85a-a4cd-4f75-a5d4-dae16b4aabde
# ╠═144027d3-d6fe-4a79-b1b8-9a5ebe4af6ee
# ╠═5108bf24-6235-4e94-9798-c954e8ae2416
# ╠═fd8b3573-48e4-4778-8d17-b2bf1971cf03
# ╠═4eacd3df-ab32-481f-bce5-a9c5576d6342
# ╠═da39a0d6-c21f-4a43-a8a8-b01761db9886
# ╠═750e2715-b8f0-4313-97ec-246223cd5c9e
# ╠═aefb8469-5386-45bf-8a37-901c5efb60e6
# ╠═7ad5b84b-02cd-4a74-9888-072be9366770
# ╟─770026ee-d572-40a0-a841-52f00cd27f75
# ╠═95d65911-1d09-47bd-9c76-1506db129901
# ╠═e543aadf-cbb4-4635-b8bf-e57241af1d8c
# ╠═4290f11d-d8c4-4a71-8b38-b1d34802818c
# ╟─8c965cbf-9b46-4d36-a7b3-11d4a82e0b72
# ╠═fe24979c-ae83-412b-a25e-2e3c3fc19c1f
# ╟─ea9d599d-db36-40ec-a958-e1972cb2f475
# ╟─d5b8ce01-3dfa-480a-8bd2-7d90c88e798a
# ╠═46e0ba73-31aa-4ae3-94d5-f988f51318d1
# ╠═7351fb16-79b9-4a48-9c11-785ec38d034c
# ╟─cf8b5fde-cba4-4f19-b14e-2fe12254a041
# ╟─54d5ae08-7443-4dfc-8859-e4233e856712
# ╠═e111dcb1-d8ff-43fb-ba54-fecf23dfd80e
# ╠═e5e4c45d-b9a9-46fe-8064-61231ca41b45
# ╠═1f31de5e-51e2-4e0e-85d4-3f3c1a6196e9
# ╠═b1006a0b-d3f1-4f20-9bb9-0f82f111c209
# ╠═9e8ab52c-bff7-4250-8a05-8d2f3b7a4c80
# ╟─d93b71e4-5882-4713-868e-241adb0681da
# ╟─4d9cf901-a42f-4807-92a3-cc183965c553
# ╠═e7c4dea7-2c2a-496f-97a1-96c045a4a6e5
# ╠═e60ae28a-6746-4aad-b128-0a75a88d9895
# ╠═4d9ea67f-6948-4e3b-9b0f-81c9207f65d5
# ╠═684eba81-0212-4943-a2dc-2ef28ebff516
# ╠═a5726855-2e54-44ea-abfc-5b0008159ef7
# ╠═ca6e34ce-4531-468c-8763-540be2b00ee7
# ╠═29ae22fd-4582-4c9c-829c-462d8dfb0286
# ╟─7418f53f-d948-4e18-b485-2213ee8dca9a
# ╠═7fef9d14-d63e-479d-a2cc-04b3194dd883
# ╠═74128555-0120-4eae-9bed-38d8e6fd58c8
# ╟─e153520d-592a-4d20-bef3-879a532a8438
# ╠═57144295-0516-4642-8066-0872840a0fca
# ╠═c2bec94f-7b7d-4b90-ba3b-1e62b36a9922
# ╟─29b05235-d45f-4919-942e-801ae4895876
# ╠═48610899-63d5-4dd5-8836-80afb4f2bc89
# ╠═837cd1f4-72d0-418e-9f45-900862e6987f
# ╠═75cfbd96-be0f-4697-93e5-cf84ec76e36d
# ╠═0dea764e-fbfa-4929-8ce5-42f0ba0cbee5
# ╠═0c35b060-bfd4-4187-87eb-91ebae01f194
# ╠═1f39ab61-7d2d-4b76-9f15-9ed0074a2339
# ╠═e39e90a8-7c95-4595-92c3-79d5d206670d
# ╟─ae51631e-9a36-4023-90d4-e85524ebb92d
# ╠═0fe8b4f8-0c48-426b-907a-a8afeb6c26bd
# ╠═80ae48ad-21bd-45c5-abcd-971e48a09d44
# ╠═65ff6161-4280-4b92-befd-0e23cd3e701b
# ╟─3b822a24-cccb-46e4-a60b-9b8a01b358d7
# ╠═64d046ac-3388-4862-835f-78437096b3f6
# ╠═c8cc8431-f4e3-46fa-a293-32dd8479782f
# ╠═1e86c61c-a2b4-4ec7-951d-a52132f96fa6
# ╠═31a28fe4-4c9b-40cf-ada1-c04a8955914f
# ╠═360f1f4d-d8a4-4502-a8b7-27f0a018da33
# ╠═0cec4057-ad39-443e-86e7-d1feebc73278
# ╠═54edc22a-99ee-4b97-902a-5c1eff374938
# ╟─05671fec-1cdd-43cc-bd5d-cc1231e9249a
# ╟─a48200da-05e4-40ad-aa5b-6622da4e1ade
# ╟─910b28ec-b2d4-4a71-9168-acf221b88b5d
# ╠═773af3b1-ee4a-4e02-9a46-cf1727567188
# ╠═b0b2201f-0193-42c2-99f6-571649051b9f
# ╠═e2fabd8b-c499-4316-aafc-cfffea951c38
# ╠═d0c67b7c-e341-4ec6-85ae-fe46240b632a
# ╠═7d350757-15fb-43db-b9ec-2c23a28bad34
# ╠═b02f73fb-7ec3-4345-a585-252de5fedc92
# ╟─d2934768-fa38-4404-b69e-f01d3bd6f39c
# ╟─94b7ba46-ce14-460f-a741-6059db8755c4
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
