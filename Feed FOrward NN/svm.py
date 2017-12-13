import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
data = scio.loadmat('HW2_Data/data1')

X_train = np.matrix(np.insert(data['X_trn'], 0, 1, axis=1))
y_train = np.matrix(data['Y_trn'])
X_test = np.matrix(np.insert(data['X_tst'], 0, 1, axis=1))
y_test = np.matrix(data['Y_tst'])

y_train = np.where(y_train == 0, -1, 1)
y_test = np.where(y_test == 0, -1, 1)
dataset = np.dot(X_train, X_train.T)

regular_param=1
tol = 0.0001 #tolerance
max_passes = 1000

#sigmoid function:
def inverse_log_odds(z):
    return 1.0 / (1.0 + np.exp(-z))

def classify(X,y,t):
    result = inverse_log_odds(X*t.T)
    for i in range(len(result)):
        if result[i] < 0.5: result[i] = 0
        else:result[i] = 1
    return result

def classificationError(y,predicted_y):
    c = 0
    for i in range(len(y)): 
        if predicted_y[i] != y[i]: c += 1
    return c

def print_error_classification(X_train,y_train,t):
    predicted_y = classify(X_train,y_train,t)
    print(predicted_y.transpose().tolist()[0])
    error = classificationError(y_train,predicted_y)
    percent_error =  error/len(predicted_y) * 100
    print("Count of miss-classified points:", error)
    print("Classification error percent: {:0.2f}%".format(percent_error))
          
    
def findEta(i,j):
    return 2 * dataset[i, j] - dataset[i, i] - dataset[j, j]

def inner_dual(X, y, b, alp, i):
    dummy = np.multiply(alp,y)
    dum2 = dummy.T * dataset[:,i] 
    return dum2 + b

def findNewAlpha(alp,y,Ei,Ej,e,L,H):
    t = (Ei-Ej)
    alp = alp - (y*t) / e
    if alp > H: return H
    elif alp < L: return L
    else: return alp

def find_b(b1,b2,ialp,jalp,regularize_param):
    if 0 < ialp and ialp < regularize_param: return b1
    elif 0 < jalp and jalp < regularize_param: return b2
    else: return (b1 + b2) / 2

# from the paper:
def smo(X,y,regular_param,tol,max_passes):
    #initialize alpha = 0; b=0;
    alp = np.matrix(np.zeros(X.shape[0]))
    alp = alp.T
    b = 0
    passes = 0
    while passes < max_passes:
        num_changed_alphas = 0
        for i in range(len(X)):
            Ei = inner_dual(X,y,b,alp,i) - y[i]
            if (y[i] * Ei < -tol and alp[i] < regular_param) or (y[i] * Ei > tol and alp[i] > 0):
                j = np.random.choice(len(X))
                # choose j!=i randmly:
                while j == i: j = np.random.choice(len(X))
                Ej = inner_dual(X,y,b,alp,j) - y[j]
                # save old alphas
                old_alpha_i = alp.item(i)
                old_alpha_j = alp.item(j)
                # • If y(i) != y(j) , L = max(0, αj − αi), H = min(C, C + αj − αi)
                if y[i] != y[j]:
                    L = max(0, alp[j] - alp[i])
                    H = min(regular_param, regular_param + alp[j] - alp[i])
                # If y(i) = y(j) , L = max(0, αi + αj − C), H = min(C, αi + αj
                else:
                    L = max(0, alp[i] + alp[j] - regular_param)
                    H = min(regular_param, alp[i] + alp[j])
                    
                if L == H: continue
                #η = 2<x(i) , x(j)> − <x(i), x(i)> − <x(j), x(j)>. 
                eta = findEta(i, j)
                if eta >= 0: continue
                    
                alp[j] = findNewAlpha(alp[j],y[j],Ei,Ej,eta,L,H)
                if abs(alp[j] - old_alpha_j) < 0.00001: continue
                # compute new value of alpha i;
                alp[i] = alp[i] + (y[i] * y[j] * (old_alpha_j - alp[j]))
                b1 = b - Ei - y[i] * (alp[i] - old_alpha_i) * dataset[i, i] - y[j] * (alp[j] - old_alpha_j) * dataset[i, j]
                b2 = b - Ej - y[i] * (alp[i] - old_alpha_i) *  dataset[i, j] - y[j] * (alp[j] - old_alpha_j) * dataset[j, j]
                # satisfy KTT condition:
                b = find_b(b1, b2, alp[i], alp[j], regular_param)
                
                num_changed_alphas += 1
        if num_changed_alphas == 0: passes += 1
        else: passes = 0
    return alp, b


alpha,b = smo(X_train, y_train, regular_param, tol, max_passes)
print("alpha:", alpha)
print("b:", b.T)
 
def graph_Plot(X, y, sl, title = 'train set'):
    X = np.array(np.concatenate((X, y), axis=1))
    X1 = X[np.ix_(X[:,3] == -1, (1,2))]
    x = np.linspace(-3, 3, 10)
    sl=np.array(sl) 
    inter_cept  = -sl[:, 0]/sl[:,2]
    m =  -sl[:,1]/sl[:,2]
    X2 = X[np.ix_(X[:,3] == 1, (1,2))]
    plt.title(title)
    plt.scatter(X2[:, 0], X2[:,1], marker='o', color="blue", label="Class:1")
    plt.scatter(X1[:, 0], X1[:,1], marker='+', color="green", label="Class:0")
      
    # y = mx+b
    y_pred = m*x + inter_cept 
    plt.plot(x, y_pred, color='blue', label="Boundary")
    m1 = np.ceil(X2.max())
    m2 = np.floor(X2.min())
    plt.ylim(m2, m1)
    plt.legend()
    plt.show()
    

dummy1 = np.matrix(np.multiply(alpha,y_train))
theta_dummy = np.dot(dummy1.T,X_train)    
graph_Plot(X_train,y_train,theta_dummy,"training set")
graph_Plot(X_test,y_test,theta_dummy,"test set")
print_error_classification(X_train, y_train, theta_dummy) 
print_error_classification(X_test, y_test, theta_dummy)