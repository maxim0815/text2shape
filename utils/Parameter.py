

class HyperParameter(object):
	"""docstring for HyperParameter"""
	def __init__(self, lr=0.001, bs=64, mom=0.9, wd=1e-4):
		super(HyperParameter, self).__init__()
		self.lr = lr
		self.bs = bs
		self.mom = mom
		self.wd = wd

	def asdict(self):
		dict_ = {'lr': self.lr, 'bs': self.bs,
				 'mom': self.mom, 'wd': self.wd}
		return dict_
		