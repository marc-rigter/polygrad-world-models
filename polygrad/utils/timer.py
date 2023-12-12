import time

class Timer:

	def __init__(self):
		self._start = time.time()
		self._step_last = 0

	def fps(self, step, reset=True):
		now = time.time()
		fps = (step - self._step_last) / (now - self._start) 
		if reset:
			self._start = now
			self._step_last = step
		return fps