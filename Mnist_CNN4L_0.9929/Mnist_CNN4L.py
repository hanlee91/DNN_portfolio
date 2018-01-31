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
        
        with tf.variable_scope(self.name):
        
            #DropOut 적용
            
            self.training = tf.placeholder(tf.bool)
            self.X = tf.placeholder(tf.float32, [None, 784], name = 'X')
            X_img = tf.reshape(self.X, [-1, 28, 28, 1])
            self.Y = tf.placeholder(tf.float32, [None, 10], name = 'Y')
            
            
            with tf.name_scope("layer1"):
                conv1 = tf.layers.conv2d(inputs=X_img, filters=64, kernel_size=[3, 3],
                                         padding="SAME", activation=tf.nn.relu)
                
                pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2],
                                                padding="SAME", strides=2)
                dropout1 = tf.layers.dropout(inputs=pool1,
                                             rate=0.7, training=self.training)
                
                
            with tf.name_scope("layer2"):
                conv2 = tf.layers.conv2d(inputs=dropout1, filters=128, kernel_size=[3, 3],
                                         padding="SAME", activation=tf.nn.relu)
                pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2],
                                                padding="SAME", strides=2)
                dropout2 = tf.layers.dropout(inputs=pool2,
                                             rate=0.7, training=self.training)
                
            with tf.name_scope("layer3"):
                conv3 = tf.layers.conv2d(inputs=dropout2, filters=256, kernel_size=[3, 3],
                                         padding="same", activation=tf.nn.relu)
                pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2],
                                                padding="same", strides=2)
                dropout3 = tf.layers.dropout(inputs=pool3,
                                             rate=0.7, training=self.training)
                
            with tf.name_scope("layer4"):
                flat = tf.reshape(dropout3, [-1, 256 * 4 * 4])
                dense4 = tf.layers.dense(inputs=flat,
                                         units=625, activation=tf.nn.relu)
                dropout4 = tf.layers.dropout(inputs=dense4,
                                             rate=0.5, training=self.training)
    
                self.logits = tf.layers.dense(inputs=dropout4, units=10)
                
        with tf.name_scope("Cost"):
            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits = self.logits, labels = self.Y))
            self.optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(self.cost)
            tf.summary.scalar("Cost", self.cost)
         
            
        with tf.name_scope("Accuracy"):
            is_correct = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
            tf.summary.scalar("Accuracy", self.accuracy)

    #학습을 실행하는 함수, DropOut적용시 keep_prop값을 기본값으로 준다 학습시엔 0.7, 예측시엔 1
    def train(self, x_data, y_data, training = True):
        return self.sess.run([self.cost, self.optimizer], feed_dict = {
            self.X: x_data, self.Y: y_data, self.training: training})
    
    #모델의 정확도를 가져오는 함수
    def get_accuracy(self, x_test, y_test, training = False):
        return self.sess.run(self.accuracy, feed_dict = {self.X: x_test, self.Y: y_test, self.training: training})
    
    #텐서보드에 표현할 그래프값을 가져오는 함수
    def get_summary(self, x_test, y_test, training = False):
        return self.sess.run(self.merged, feed_dict = {self.X: x_test, self.Y: y_test, self.training: training})

    #예측값 리턴 함수
    def predict(self, x_test, training = False):
        return self.sess.run(self.logits, feed_dict = {self.X: x_test, self.training: training})
    
tf.set_random_seed(777)

learning_rate = 0.001
training_epochs = 100
batch_size = 100

#데이터셋 가져오기
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#세션 생성
sess = tf.Session()

#모델 생성
model = Model(sess, "model")

#텐서보드 FileWriter 생성
writer = tf.summary.FileWriter("C:/data/logs/Mnist_CNN4L", sess.graph)

#세션 초기화
sess.run(tf.global_variables_initializer())

#학습시간 측정
tr_data = []
s_time = time.time()


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
path = 'C:/Users/HAN/Desktop/DL_P_D/Mnist_CNN4L' + str(acr) + '/Mnist_CNN4L' + str(acr) + '.pd'
save_path = saver.save(sess, path)
print ('모델 저장경로',save_path)

#숫자 이미지를 랜덤하게 추출하여 예측 결과를 보여
def visualtest():
    r = random.randint(0, len(mnist.test.labels) - 1)
    a = np.array(mnist.test.images[r:r + 1])
    plt.imshow(a.reshape(28, 28))
    pre = model.predict(mnist.test.images[r:r + 1])
    lbl = mnist.test.labels[r:r + 1]
    print("예측값:", list(pre[0]).index(max(pre[0])), "실제값:", list(lbl[0]).index(max(lbl[0])))

visualtest()
