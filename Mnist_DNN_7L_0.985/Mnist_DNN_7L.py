import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt
import time
from tensorflow.examples.tutorials.mnist import input_data


class Model:

    #초기화
    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self.build_net()
        self.merged = tf.summary.merge_all()
    
    #변수생성 함수
    def init_var(shape, name):
        return tf.get_variable(shape = shape, name = name, initializer = tf.contrib.layers.xavier_initializer())
    
    #네트워크 구성 함수
    def build_net(self):
        
        #7층의 Layer로 구성
        self.X = tf.placeholder(tf.float32, [None, 784], name = 'X')
        self.Y = tf.placeholder(tf.float32, [None, 10], name = 'Y')
        
        
        with tf.name_scope("layer1"):
            W1 = Model.init_var([784, 512], "W1")
            b1 = Model.init_var([512], "b1")
            #TensorBoard에서 구현할 histogram값을 선언
            tf.summary.histogram("W1_Values", W1)
            tf.summary.histogram("B1_Values", b1)
            L1 = tf.nn.relu(tf.matmul(self.X, W1) + b1)
            
        with tf.name_scope("layer2"):
            W2 = Model.init_var([512, 512], "W2")
            b2 = Model.init_var([512], "b2")
            tf.summary.histogram("W2_Values", W2)
            tf.summary.histogram("B2_Values", b2)
            L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
            
        with tf.name_scope("layer3"):
            W3 = Model.init_var([512, 512], "W3")
            b3 = Model.init_var([512], "b3")
            tf.summary.histogram("W3_Values", W3)
            tf.summary.histogram("B3_Values", b3)
            L3 = tf.nn.relu(tf.matmul(L2, W3) + b3)
            
        with tf.name_scope("layer4"):
            W4 = Model.init_var([512, 512], "W4")
            b4 = Model.init_var([512], "b4")
            tf.summary.histogram("W4_Values", W4)
            tf.summary.histogram("B4_Values", b4)
            L4 = tf.nn.relu(tf.matmul(L3, W4) + b4)
            
        with tf.name_scope("layer5"):
            W5 = Model.init_var([512, 512], "W5")
            b5 = Model.init_var([512], "b5")
            tf.summary.histogram("W5_Values", W5)
            tf.summary.histogram("B5_Values", b5)
            L5 = tf.nn.relu(tf.matmul(L4, W5) + b5)
            
        with tf.name_scope("layer6"):
            W6 = Model.init_var([512, 512], "W6")
            b6 = Model.init_var([512], "b6")
            tf.summary.histogram("W6_Values", W6)
            tf.summary.histogram("B6_Values", b6)
            L6 = tf.nn.relu(tf.matmul(L5, W6) + b6)
            
        with tf.name_scope("layer7"):
            W7 = Model.init_var([512, 10], "W7")
            b7 = Model.init_var([10], "b7")
            tf.summary.histogram("W7_Values", W7)
            tf.summary.histogram("B7_Values", b7)
            self.hypothesis = tf.matmul(L6, W7) + b7
            
        with tf.name_scope("Cost"):
            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits = self.hypothesis, labels = self.Y))
            self.optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(self.cost)
            tf.summary.scalar("Cost", self.cost)
         
            
        with tf.name_scope("Accuracy"):
            is_correct = tf.equal(tf.argmax(self.hypothesis, 1), tf.argmax(self.Y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
            tf.summary.scalar("Accuracy", self.accuracy)

    #학습을 실행하는 함수
    def train(self, x_data, y_data):
        return self.sess.run([self.cost, self.optimizer], feed_dict = {
            self.X: x_data, self.Y: y_data})
    
    #모델의 정확도를 가져오는 함수
    def get_accuracy(self, x_test, y_test):
        return self.sess.run(self.accuracy, feed_dict = {self.X: x_test, self.Y: y_test})
    
    #텐서보드에 표현할 그래프값을 가져오는 함수
    def get_summary(self, x_test, y_test):
        return self.sess.run(self.merged, feed_dict = {self.X: x_test, self.Y: y_test})

    #예측값 리턴 함수
    def predict(self, x_test):
        return self.sess.run(self.hypothesis, feed_dict = {self.X: x_test})
    
tf.set_random_seed(777)

learning_rate = 0.001
training_epochs = 100
batch_size = 100

#데이터셋 가져오기
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#세션 생성
sess = tf.Session()

#학습시간 측정
tr_data = []
s_time = time.time()

#모델 생성
model = Model(sess, "model")

#텐서보드 FileWriter 생성
writer = tf.summary.FileWriter("C:/data/logs/Mnist_DNN_7L", sess.graph)

#세션 초기화
sess.run(tf.global_variables_initializer())

print('학습 시작!')

for epoch in range(training_epochs):
    e_s_time = time.time()
    avg_cost = 0
    
    #미니배치 사용
    total_batch = int(mnist.train.num_examples / batch_size)
    
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        c, _ = model.train(batch_xs, batch_ys)
        avg_cost += c / total_batch
        
    #현재 Epoch의 상태를 텐서보드Log파일에 쓰기
    summary = model.get_summary(mnist.test.images, mnist.test.labels)
    writer.add_summary(summary, epoch + 1)
    
    #현재 Epoch의 상태를 출력
    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost), '학습시간 : ' + str(round(time.time() - e_s_time, 3)) + '초')

    #현재 Epoch의 학습시간을 저장
    tr_data.append(str(epoch + 1) + '에폭 학습시간 : ' + str(round(time.time() - e_s_time, 3)) + '초')

#총 학습시간을 저장
tr_data.append('총 학습시간 : ' + str(round(time.time() - s_time, 3)) + '초')
#for i in tr_data:
#    print(i)
    
writer.close()
print('학습 종료!')

#모델의 정확도를 출력
acr = model.get_accuracy(mnist.test.images, mnist.test.labels)
print('정확도 : ' + str(acr) + ' 총 학습시간 : ' + str(round(time.time() - s_time, 3)) + '초')

#모델 저장
saver = tf.train.Saver()
path = 'C:/Users/HAN/Desktop/DL_P_D/Mnist_DNN_7L' + str(acr) + '/Mnist_DNN_7L' + str(acr) + '.pd'
save_path = saver.save(sess, path)
print ('모델 저장경로',save_path)

#숫자 이미지를 랜덤하게 추출하여 예측 결과를 보여
def visualtest():
    r = random.randint(0, len(mnist.test.labels) - 1)
    pre = model.predict(mnist.test.images[r:r + 1])
    lbl = mnist.test.labels[r:r + 1]
    a = np.array(mnist.test.images[r:r + 1])
    plt.imshow(a.reshape(28, 28))
    print("예측값:", list(pre[0]).index(max(pre[0])), "실제값:", list(lbl[0]).index(max(lbl[0])))

visualtest()