import pickle
import gzip

with gzip.open('/mnt_pool/c3_users/user/jriego/tchannel5TeV/cafea/retaking/TTPS_part0.pkl.gz','rb') as file:
	loaded_data=pickle.load(file)

print(loaded_data)
