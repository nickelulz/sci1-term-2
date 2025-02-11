def Calix(output_data_sequence, markov, n, delta=0.01):
  #n = number of outputs
  estimated_model = np.random.rand(n, n) * 100
  observations = np.zeros(shape=(n,n))
  cur = 0 #start storing values in the first coin by default
  epoch=0
  for obs in output_data_sequence:
    i = markov[cur].index(obs)
    observations[cur][i] += 1
    cur = i
    row_sums = observations.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    gradients = (observations / row_sums)*100 - estimated_model
    estimated_model += delta * gradients
    if epoch % 2500 == 0:
      print(f"Cur Est: {estimated_model}")
      print(f"Cur Obs: {observations}")
      print(f"Obs Markov: {(observations / row_sums)*100}")
    epoch+=1
  return CoinFlipEngine(
    probabilities = estimated_model,
    markov = np.array(markov),
    number_of_coins = n,
    number_of_outputs = n,
    initial_coin_index = 0)
