Had to update the shifterator package as it is outdated.

in in shifterator/plotting.py on line 745 changed 

        "tight": True,
        
        to

        "tight": False,


in shifterator/plotting.py on line 745 changed 
'''for tick in in_ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(11)'''
    
to 

'''in_ax.tick_params(axis='y', labelsize=11)'''


in shifterator/plotting.py on line 799 changed 

  '''for tick in in_ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(12)'''
to
    
  '''in_ax.tick_params(axis='y', labelsize=12)'''

in shifterator/helper py on line 157 changed 

'''if isinstance(scores, collections.Mapping):
        return scores.copy(), None'''
to

'''if isinstance(scores, collections.abc.Mapping):
        return scores.copy(), None'''
