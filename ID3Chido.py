from id3 import Id3Estimator, export_graphviz
import numpy as np
import graphviz

features = ["Categories", "Genres"]
##datatable
x = np.array([['Multi-player;Co-op;Steam Trading Cards;Steam Workshop;SteamVR Collectibles;In-App Purchases;Valve Anti-Cheat enabled',
  'Action;Free to Play;Strategy'],
 ['Single-player;Co-op;Steam Achievements;Full controller support;Steam Trading Cards;Captions available;Steam Workshop;Steam Cloud;Stats;Includes level editor;Commentary available',
  'Action;Adventure'],
 ['Single-player;Multi-player;Co-op;Steam Achievements;Captions available;Steam Cloud;Stats;Includes level editor',
  'Action'],
 ['Multi-player;Steam Achievements;Full controller support;Steam Trading Cards;Steam Workshop;In-App Purchases;Valve Anti-Cheat enabled;Stats',
  'Action;Free to Play'],
 ['Single-player;Multi-player', 'Indie'],
 ['Multi-player;Steam Achievements;Valve Anti-Cheat enabled', 'Action'],
 ['Single-player;Multi-player;Co-op;Cross-Platform Multiplayer;Steam Achievements;Steam Trading Cards;Steam Workshop;Valve Anti-Cheat enabled;Stats;Includes level editor',
  'Action'],
 ['Single-player;Stats', 'Action'],
 ['Single-player', 'Indie;Strategy'],
 ['Single-player', 'Indie;Strategy'],
 ['Single-player;Multi-player', 'Indie;Strategy'],
 ['Single-player;Multi-player;Steam Achievements', 'Indie'],
 ['Single-player;Multi-player', 'Strategy'],
 ['Single-player;Multi-player', 'Strategy'],
 ['Single-player;Multi-player;Co-op', 'Strategy'],
 ['Single-player;Multi-player;Co-op', 'Strategy'],
 ['Single-player;Multi-player', 'Strategy'],
 ['Single-player;Multi-player', 'Strategy'],
 ['Single-player', 'RPG'],
 ['Steam Workshop', 'Animation & Modeling;Video Production'],
 ['Single-player;Multi-player;Co-op;Cross-Platform Multiplayer;Steam Trading Cards',
  'Strategy'],
 ['Single-player;Multi-player;Online Multi-Player;Online Co-op;Cross-Platform Multiplayer;Steam Trading Cards;Partial Controller Support',
  'RPG'],
 ['Single-player;Multi-player;Valve Anti-Cheat enabled', 'Action;RPG'],
 ['Single-player;Multi-player;Steam Cloud', 'Action'],
 ['Single-player;Multi-player;Steam Cloud', 'Action'],
 ['Single-player;Steam Cloud', 'Action'],
 ['Single-player;Steam Cloud', 'Action'],
 ['Single-player;Steam Cloud', 'Action']])


print(type(x))

y = np.array(['Valve', 'Valve', 'Valve', 'Valve',
 'Mark Healey', 'Tripwire Interactive', 'Tripwire Interactive',
 'Ritual Entertainment', 'Introversion Software', 'Introversion Software',
 'Introversion Software', 'Introversion Software', 'Strategy First',
 'Strategy First', 'Strategy First', 'Strategy First', 'Strategy First', 
 'Strategy First', 'Arkane Studios', 'Valve', 'Topware Interactive;ACE',
 'Topware Interactive', 'Ubisoft', 'id Software', 'Bethesda Softworks',
 'Bethesda-Softworks', 'id Software', 'id Software'])

#print(y.shape)

# Leave-one-out#########################################################
n = np.random.randint(0, x.shape[0],)## select randomly a data test
#print(x.shape[0],)
#print(n)
x_test = np.array([x[n],],)
#print(x_test)
x_train = np.delete(x, n, 0)
print(type(x_train))
y_test = np.array([y[n],],)
y_train = np.delete(y, n, 0)
########################################################################

id3 = Id3Estimator()####################################################
id3.fit(x_train, y_train)###############################################

# Testing

y_predict = id3.predict(x_test)#########################################
# Precision
print("Precision")
print("Input: ", x_test, "| Expected: ", y_test, "| Result: ", y_predict)
if np.array_equal(y_test, y_predict):
    print("100%")
else:
    print("0%")


export_graphviz(id3.tree_, 'tree_p1.dot', features)
with open("tree_p1.dot") as f:
    dot_graph = f.read()
g = graphviz.Source(dot_graph)
g.render()
g.view()