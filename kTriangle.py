##Building a momentum (k) triangle that only allows for closed k-space configurations in bispectrum
import sys
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import pandas as pd
from math import e
from scipy.interpolate import interp2d, interp1d,InterpolatedUnivariateSpline,RectBivariateSpline
from global_variables import * 
class kTriangle(object):
	def __init__(self,in1,in2,in3,input="SAS"):
		"""if using SAS, then, in1=length of k1, in2=length of k2, in3 = angle in radians between k1 and k2"""
		"""if using SSS, then, in1=length of k1, in2=length of k2, in3 = length of k3"""
		if input == "SAS":
			self.k1 = in1
			self.k2 = in2
			self.cos12 = np.cos(in3)
			self.k3 = np.sqrt(self.k1**2 + self.k2**2 + 2*self.k1*self.k2*self.cos12)

		elif input == "SSS":
			self.k1 = in1
			self.k2 = in2
			self.k3 = in3
			self.cos12 = (-self.k1**2 - self.k2**2 + self.k3**2)/(2*self.k1*self.k2)
		
		
		self.cos13 = (-self.k1**2 - self.k3**2 + self.k2**2)/(2*self.k1*self.k3)
		self.cos23 = (-self.k2**2 - self.k3**2 + self.k1**2)/(2*self.k2*self.k3)

		assert self.k1 + self.k2 >= self.k3
	"""output for SAS method, length of k1, length of k2, and cosine of angle between k1 and k2"""
	def output_SAS(self):
		return (self.k1,self.k2,self.cos12)
	"""output for SSS method, length of k1, length of k2, length of k3"""
	def output_SSS(self):
		return (self.k1,self.k2,self.k3)