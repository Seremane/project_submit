# -*- coding: utf-8 -*-
"""
Created on Sun Jun 12 13:42:21 2016

@author: ontibile
"""

import numpy as np
from matplotlib import pyplot as plt
from numpy.fft import fft, ifft, fftfreq
import time


class Particles:
    def __init__(self,n=1000,G=1.0,soft=0.1,m=1.0,dt=0.1,ncell=500):
        self.val={}
        self.val['n']=n
        self.val['dt']=dt
        self.val['G']=G
        
        self.den=np.ones(self.val['n'])
        self.x=np.random.randn(n)
        self.y=np.random.randn(n)
        self.m=np.ones(self.val['n'])
        self.vx=0*self.x
        self.vy=0*self.x
        #density of each particle
        #self.v=numpy.ones(self.val['n'])
        self.val['soft']=soft
        self.ncell=ncell

    def softened_pot(self):
        pot=0
        for i in range(0,self.opts['n']-1):
            dx=self.x[i]-self.x[i+1:]
            dy=self.y[i]-self.y[i+1:]
            rsqr=(dx*dx+dy*dy)
            soft=self.val['soft']**2
            rsqr[rsqr<soft]=soft
            rsqr=rsqr+soft
            r=np.sqrt(rsqr)
            r3inv=1.0/(r*rsqr)
            self.fx[i]=-np.sum(self.m*dx*r3)*self.val['G']
            self.fy[i]=-np.sum(self.m*dy*r3)*self.val['G']
            pot+=self.val['G']*np.sum(self.m/r)
        return -0.5*pot
    
    
    def rho_den(self):
        rho=0.0
        number=1
        m=self.m
        x=self.x
        y=self.y
        for i in range(len(x)):
            r_sq= (self.x[i]-self.x)**2+(self.y[i]-self.y)**2
            if (r_sq<number):
                if (r_sq<0.5*qs):
                    rho += 8.0/np.pi*(1.0-6.0*r_sq**2 + 0.6*r_sq**3)
                else:
                        rho += 16.0/np.pi* (1.0-r_sq)**3
        return rho*m

    def potential(self):
        density=self.rho_den()
        soft_pot=self.softened_pot()
        ft1=fft(density)
        ft2=fft(soft_pot)
        conv_pot=ifft(ft1*ft2)
        return conv_pot
    
    
    def evolve(self):
        self.x+=self.vx*self.val['dt']
        self.y+=self.vy*self.val['dt']
        pot=self.potential()
        self.vx+=self.fx*self.val['dt']
        self.vy+=self.fy*self.val['dt']
        kinetic=0.5*np.sum(self.m*(self.vx**2+self.vy**2))
        return pot+kinetic    
        

if __name__=='__main__':
    plt.ion
    nn=3000
    particle=Particles()
    plt.clf()
    plt.plot(particle.x,particle.y,'*')
    plt.show()