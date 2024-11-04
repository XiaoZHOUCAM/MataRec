from tensorflow.keras.layers import Dense, Activation, Embedding
from tensorflow.keras.layers import Input, Flatten, dot, concatenate, Dropout
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras import initializers
import tensorflow as tf


class SurroundingSlots(tf.keras.layers.Layer):
    def __init__(self, window_length, max_range, trainable=True, name=None, **kwargs):
        super(SurroundingSlots, self).__init__(name=name, trainable=trainable, **kwargs)
        self.window_length = window_length
        self.max_range = max_range

    def build(self, inshape):
        1

    def call(self, x):
        surr = K.cast(x, dtype=tf.int32) + K.arange(start=-self.window_length, stop=self.window_length + 1, step=1)
        surrUnderflow = K.cast(surr < 0, dtype=tf.int32)
        surrOverflow = K.cast(surr > self.max_range - 1, dtype=tf.int32)

        return surr * (-(surrUnderflow + surrOverflow) + 1) + surrUnderflow * (surr + self.max_range) + surrOverflow * (
                    surr - self.max_range)

    def compute_output_shape(self, inshape):
        return (inshape[0], self.window_length * 2 + 1)


class PTE(tf.keras.layers.Layer):
    def __init__(self, dimension, trainable=True, name=None, **kwargs):
        super(MATE, self).__init__(name=name, trainable=trainable, **kwargs)
        self.dimension = dimension

    def build(self, inshape):
        # for multiplicative attention
        self.W = self.add_weight(name="W", shape=(self.dimension, self.dimension),
                                 initializer=initializers.get("random_normal"))

        # for personalization
        self.Wmonth = self.add_weight(name="Wmonth", shape=(self.dimension, self.dimension),
                                      initializer=initializers.get("random_normal"))
        self.Wday = self.add_weight(name="Wday", shape=(self.dimension, self.dimension),
                                    initializer=initializers.get("random_normal"))
        self.Wdate = self.add_weight(name="Wdate", shape=(self.dimension, self.dimension),
                                     initializer=initializers.get("random_normal"))
        self.Whour = self.add_weight(name="Whour", shape=(self.dimension, self.dimension),
                                     initializer=initializers.get("random_normal"))

    def call(self, x):
        userEmbedding = x[0]

        curMonthEmbedding = K.reshape(x[1], shape=(-1, 1, self.dimension))
        curDayEmbedding = K.reshape(x[2], shape=(-1, 1, self.dimension))
        curDateEmbedding = K.reshape(x[3], shape=(-1, 1, self.dimension))
        curHourEmbedding = K.reshape(x[4], shape=(-1, 1, self.dimension))

        monthEmbeddings = x[5]
        dayEmbeddings = x[6]
        dateEmbeddings = x[7]
        hourEmbeddings = x[8]

        # personalization
        curMonthEmbedding = curMonthEmbedding * (K.dot(userEmbedding, self.Wmonth))
        curDayEmbedding = curDayEmbedding * (K.dot(userEmbedding, self.Wday))
        curDateEmbedding = curDateEmbedding * (K.dot(userEmbedding, self.Wdate))
        curHourEmbedding = curHourEmbedding * (K.dot(userEmbedding, self.Whour))
        monthEmbeddings = monthEmbeddings * (K.dot(userEmbedding, self.Wmonth))
        dayEmbeddings = dayEmbeddings * (K.dot(userEmbedding, self.Wday))
        dateEmbeddings = dateEmbeddings * (K.dot(userEmbedding, self.Wdate))
        hourEmbeddings = hourEmbeddings * (K.dot(userEmbedding, self.Whour))

        # query for gradated attention
        monthQ = curMonthEmbedding
        dayQ = curDayEmbedding
        dateQ = curDateEmbedding
        hourQ = curHourEmbedding

        #         key, value
        monthKV = concatenate([monthEmbeddings, curMonthEmbedding], axis=1)
        dayKV = concatenate([dayEmbeddings, curDayEmbedding], axis=1)
        dateKV = concatenate([dateEmbeddings, curDateEmbedding], axis=1)
        hourKV = concatenate([hourEmbeddings, curHourEmbedding], axis=1)

        # attention score
        monthQKV = K.softmax(K.batch_dot(monthQ, K.permute_dimensions(monthKV, pattern=(0, 2, 1))) / K.sqrt(
            K.cast(self.dimension, dtype=tf.float32)), axis=-1)
        dayQKV = K.softmax(K.batch_dot(dayQ, K.permute_dimensions(dayKV, pattern=(0, 2, 1))) / K.sqrt(
            K.cast(self.dimension, dtype=tf.float32)), axis=-1)
        dateQKV = K.softmax(K.batch_dot(dateQ, K.permute_dimensions(dateKV, pattern=(0, 2, 1))) / K.sqrt(
            K.cast(self.dimension, dtype=tf.float32)), axis=-1)
        hourQKV = K.softmax(K.batch_dot(hourQ, K.permute_dimensions(hourKV, pattern=(0, 2, 1))) / K.sqrt(
            K.cast(self.dimension, dtype=tf.float32)), axis=-1)

        # embedding for each granularity of period information
        monthEmbedding = K.batch_dot(monthQKV, monthKV)
        dayEmbedding = K.batch_dot(dayQKV, dayKV)
        dateEmbedding = K.batch_dot(dateQKV, dateKV)
        hourEmbedding = K.batch_dot(hourQKV, hourKV)

        # multiplicative attention
        q = userEmbedding
        kv = K.concatenate([monthEmbedding, dayEmbedding, dateEmbedding, hourEmbedding], axis=1)
        qW = K.dot(q, self.W)
        a = K.sigmoid(K.batch_dot(qW, K.permute_dimensions(kv, pattern=(0, 2, 1))))
        timeRepresentation = K.batch_dot(a, kv)
        return timeRepresentation

    def compute_output_shape(self, inshape):
        return (None, 1, self.dimension)


class CTHE(tf.keras.layers.Layer):
    def __init__(self, dimension, trainable=True, name=None, **kwargs):
        super(TAHE, self).__init__(name=name, trainable=trainable, **kwargs)
        self.dimension = dimension

    def build(self, inshape):
        1

    def call(self, x):
        recentTimeRepresentations = x[0]
        curTimeRepresentation = x[1]
        recentTimestamps = x[2]
        recentItemEmbeddings = x[3]

        # previous timestamp == 0 ==> no history
        mask = K.cast(recentTimestamps > 0, dtype=tf.float32)

        # time-based attention
        similarity = K.batch_dot(K.l2_normalize(curTimeRepresentation, axis=-1),
                                 K.permute_dimensions(K.l2_normalize(recentTimeRepresentations, axis=-1),
                                                      pattern=(0, 2, 1)))
        #         masked_similarity = mask * ((similarity + 1.0) / 2.0)
        #         print(masked_similarity)
        #         print(recentItemEmbeddings)
        weightedPrevItemEmbeddings = K.batch_dot(similarity, recentItemEmbeddings)
        userHistoryRepresentation = weightedPrevItemEmbeddings

        return userHistoryRepresentation

    def compute_output_shape(self, inshape):
        return (None, self.dimension)


class meanLayer(tf.keras.layers.Layer):
    def __init__(self, trainable=True, name=None, **kwargs):
        super(meanLayer, self).__init__(name=name, trainable=trainable, **kwargs)

    def build(self, inshape):
        1

    def call(self, x):
        return K.mean(x, axis=1, keepdims=True)

    def compute_output_shape(self, inshape):
        return (inshape[0], 1, inshape[2])


class Slice(tf.keras.layers.Layer):
    def __init__(self, index, trainable=True, name=None, **kwargs):
        super(Slice, self).__init__(name=name, trainable=trainable, **kwargs)
        self.index = index

    def build(self, inshape):
        1

    def call(self, x):
        return x[:, self.index, :]

    def compute_output_shape(self, inshape):
        return (inshape[0], inshape[2])


def MataRec(input_shape, num_users, num_items, embedding_size, sequence_length, popular_length, width, depth, dropout_rate=None, features=False):
    sexInput = tf.keras.Input(shape=[1], dtype=tf.int32)
    ageInput = tf.keras.Input(shape=[1], dtype=tf.int32)
    brandInput = tf.keras.Input(shape=[1], dtype=tf.int32)
    provinceInput = tf.keras.Input(shape=[1], dtype=tf.int32)
    userInput = tf.keras.Input(shape=[1], dtype=tf.int32)
    itemInput = tf.keras.Input(shape=[1], dtype=tf.int32)
    monthInput = tf.keras.Input(shape=[1], dtype=tf.int32)
    dayInput = tf.keras.Input(shape=[1], dtype=tf.int32)
    dateInput = tf.keras.Input(shape=[1], dtype=tf.int32)
    hourInput = tf.keras.Input(shape=[1], dtype=tf.int32)
    curTimestampInput = tf.keras.Input(shape=[1], dtype=tf.int32)


    recentMonthInput = []
    recentDayInput = []
    recentDateInput = []
    recentHourInput = []
    recentTimestampInput = []
    recentItemidInput = []
    for i in range(sequence_length):
        recentMonthInput.append(tf.keras.Input(shape=[1], dtype=tf.int32))
    for i in range(sequence_length):
        recentDayInput.append(tf.keras.Input(shape=[1], dtype=tf.int32))
    for i in range(sequence_length):
        recentDateInput.append(tf.keras.Input(shape=[1], dtype=tf.int32))
    for i in range(sequence_length):
        recentHourInput.append(tf.keras.Input(shape=[1], dtype=tf.int32))
    for i in range(sequence_length):
        recentTimestampInput.append(tf.keras.Input(shape=[1], dtype=tf.int32))
    for i in range(sequence_length):
        recentItemidInput.append(tf.keras.Input(shape=[1], dtype=tf.int32))

    hotMonthInput = []
    hotDayInput = []
    hotDateInput = []
    hotHourInput = []
    for i in range(popular_length):
        hotMonthInput.append(tf.keras.Input(shape=[1], dtype=tf.int32))
    for i in range(popular_length):
        hotDayInput.append(tf.keras.Input(shape=[1], dtype=tf.int32))
    for i in range(popular_length):
        hotDateInput.append(tf.keras.Input(shape=[1], dtype=tf.int32))
    for i in range(popular_length):
        hotHourInput.append(tf.keras.Input(shape=[1], dtype=tf.int32))

    userEmbedding = tf.keras.layers.Embedding(num_users + 1, embedding_size)(userInput)
    itemEmbeddingSet = tf.keras.layers.Embedding(num_items + 1, embedding_size)
    itemEmbedding = itemEmbeddingSet(itemInput)
    recentItemEmbeddings = itemEmbeddingSet(tf.keras.layers.concatenate(recentItemidInput, axis=-1))
    recentTimestamps = tf.keras.layers.concatenate(recentTimestampInput, axis=-1)

    monthEmbedding = tf.keras.layers.Embedding(12, embedding_size)
    dayEmbedding = tf.keras.layers.Embedding(7, embedding_size)
    dateEmbedding = tf.keras.layers.Embedding(31, embedding_size)
    hourEmbedding = tf.keras.layers.Embedding(24, embedding_size)

    if features:
        sexEmbedding = tf.keras.layers.Embedding(2, embedding_size)(sexInput)
        ageEmbedding = tf.keras.layers.Embedding(85, embedding_size)(ageInput)

        brandEmbedding = tf.keras.layers.Embedding(76, embedding_size)(brandInput)
        provinceEmbedding = tf.keras.layers.Embedding(88, embedding_size)(provinceInput)
        userfeature = sexEmbedding + ageEmbedding #+ brandEmbedding
        userEmbedding = (sexEmbedding + ageEmbedding + brandEmbedding + provinceEmbedding)/4
        userfVector = Flatten()(userfeature)

    curMonthEmbedding = monthEmbedding(monthInput)
    curDayEmbedding = dayEmbedding(dayInput)
    curDateEmbedding = dateEmbedding(dateInput)
    curHourEmbedding = hourEmbedding(hourInput)

    recentMonthEmbeddings = monthEmbedding(concatenate(recentMonthInput, axis=-1))
    recentDayEmbeddings = dayEmbedding(concatenate(recentDayInput, axis=-1))
    recentDateEmbeddings = dateEmbedding(concatenate(recentDateInput, axis=-1))
    recentHourEmbeddings = hourEmbedding(concatenate(recentHourInput, axis=-1))

    hotMonthEmbeddings = monthEmbedding(concatenate(hotMonthInput, axis=-1))
    hotDayEmbeddings = dayEmbedding(concatenate(hotDayInput, axis=-1))
    hotDateEmbeddings = dateEmbedding(concatenate(hotDateInput, axis=-1))
    hotHourEmbeddings = hourEmbedding(concatenate(hotHourInput, axis=-1))

    monthEmbeddings = []
    dayEmbeddings = []
    dateEmbeddings = []
    hourEmbeddings = []

    prevMonthEmbeddings = []
    prevDayEmbeddings = []
    prevDateEmbeddings = []
    prevHourEmbeddings = []

    hprevMonthEmbeddings = []
    hprevDayEmbeddings = []
    hprevDateEmbeddings = []
    hprevHourEmbeddings = []

    ratio = 0.2
    for i in range(sequence_length):
        prevMonthEmbeddings.append([])
        monthSurr = monthEmbedding(SurroundingSlots(window_length=1, max_range=12)(recentMonthInput[i]))
        prevMonthEmbeddings[i].append(meanLayer()(monthSurr))

        prevDayEmbeddings.append([])
        daySurr = dayEmbedding(SurroundingSlots(window_length=1, max_range=7)(recentDayInput[i]))
        prevDayEmbeddings[i].append(meanLayer()(daySurr))

        prevDateEmbeddings.append([])
        dateSurr = dateEmbedding(SurroundingSlots(window_length=1, max_range=31)(recentDateInput[i]))
        prevDateEmbeddings[i].append(meanLayer()(dateSurr))

        prevHourEmbeddings.append([])
        hourSurr = hourEmbedding(SurroundingSlots(window_length=1, max_range=24)(recentHourInput[i]))
        prevHourEmbeddings[i].append(meanLayer()(hourSurr))

    monthSurr = monthEmbedding(SurroundingSlots(window_length=1, max_range=12)(monthInput))
    monthEmbeddings.append(meanLayer()(monthSurr))
    daySurr = dayEmbedding(SurroundingSlots(window_length=1, max_range=7)(dayInput))
    dayEmbeddings.append(meanLayer()(daySurr))
    dateSurr = dateEmbedding(SurroundingSlots(window_length=1, max_range=31)(dateInput))
    dateEmbeddings.append(meanLayer()(dateSurr))
    hourSurr = hourEmbedding(SurroundingSlots(window_length=1, max_range=24)(hourInput))
    hourEmbeddings.append(meanLayer()(hourSurr))

    monthEmbeddings = monthEmbeddings[0]
    for i in range(sequence_length):
        prevMonthEmbeddings[i] = prevMonthEmbeddings[i][0]

    dayEmbeddings = dayEmbeddings[0]
    for i in range(sequence_length):
        prevDayEmbeddings[i] = prevDayEmbeddings[i][0]

    dateEmbeddings = dateEmbeddings[0]
    for i in range(sequence_length):
        prevDateEmbeddings[i] = prevDateEmbeddings[i][0]

    hourEmbeddings = hourEmbeddings[0]
    for i in range(sequence_length):
        prevHourEmbeddings[i] = prevHourEmbeddings[i][0]

    ####
    for i in range(popular_length):
        hprevMonthEmbeddings.append([])
        monthSurr = monthEmbedding(SurroundingSlots(window_length=1, max_range=12)(hotMonthInput[i]))
        hprevMonthEmbeddings[i].append(meanLayer()(monthSurr))

        hprevDayEmbeddings.append([])
        daySurr = dayEmbedding(SurroundingSlots(window_length=1, max_range=7)(hotDayInput[i]))
        hprevDayEmbeddings[i].append(meanLayer()(daySurr))

        hprevDateEmbeddings.append([])
        dateSurr = dateEmbedding(SurroundingSlots(window_length=1, max_range=31)(hotDateInput[i]))
        hprevDateEmbeddings[i].append(meanLayer()(dateSurr))

        hprevHourEmbeddings.append([])
        hourSurr = hourEmbedding(SurroundingSlots(window_length=1, max_range=24)(hotHourInput[i]))
        hprevHourEmbeddings[i].append(meanLayer()(hourSurr))

    for i in range(popular_length):
        hprevMonthEmbeddings[i] = hprevMonthEmbeddings[i][0]

    for i in range(popular_length):
        hprevDayEmbeddings[i] = hprevDayEmbeddings[i][0]

    for i in range(popular_length):
        hprevDateEmbeddings[i] = hprevDateEmbeddings[i][0]

    for i in range(popular_length):
        hprevHourEmbeddings[i] = hprevHourEmbeddings[i][0]


    userVector = Flatten()(userEmbedding)
    itemVector = Flatten()(itemEmbedding)

    curTimeRepresentation = Flatten()(PTE(embedding_size)(
        [userEmbedding, curMonthEmbedding, curDayEmbedding, curDateEmbedding, curHourEmbedding, monthEmbeddings,
         dayEmbeddings, dateEmbeddings, hourEmbeddings]))  # None * embedding_size
    prevTimeRepresentations = []
    for i in range(sequence_length):
        prevTimeRepresentations.append(PTE(embedding_size)(
            [userEmbedding, Slice(i)(recentMonthEmbeddings), Slice(i)(recentDayEmbeddings),
             Slice(i)(recentDateEmbeddings), Slice(i)(recentHourEmbeddings), prevMonthEmbeddings[i],
             prevDayEmbeddings[i], prevDateEmbeddings[i], prevHourEmbeddings[i]]))  # None * embedding_size)
    prevTimeRepresentations = concatenate(prevTimeRepresentations, axis=1)

    hprevTimeRepresentations = []
    for i in range(5):
        hprevTimeRepresentations.append(PTE(embedding_size)(
            [userEmbedding, Slice(i)(hotMonthEmbeddings), Slice(i)(recentDayEmbeddings), Slice(i)(hotDateEmbeddings),
             Slice(i)(hotHourEmbeddings), hprevMonthEmbeddings[i], hprevDayEmbeddings[i], hprevDateEmbeddings[i],
             hprevHourEmbeddings[i]]))  # None * embedding_size)
    hprevTimeRepresentations = (hprevTimeRepresentations[0] + hprevTimeRepresentations[1] + hprevTimeRepresentations[
        2] + hprevTimeRepresentations[3] + hprevTimeRepresentations[4]) / 5

    transformer_features = recentItemEmbeddings
    attention_output = MultiHeadAttention(
        n_heads=1, head_dim=transformer_features.shape[2], dropout_rate=dropout_rate
    )([transformer_features, transformer_features, transformer_features])

    # Transformer block.
    attention_output = tf.keras.layers.Dropout(dropout_rate)(attention_output)
    attention_output = tf.keras.layers.Dense(embedding_size)(attention_output)
    x1 = tf.keras.layers.Add()([transformer_features, attention_output])
    x1 = tf.keras.layers.LayerNormalization()(x1)
    x2 = tf.keras.layers.LeakyReLU()(x1)
    x2 = tf.keras.layers.Dense(units=x2.shape[-1])(x2)
    x2 = tf.keras.layers.Dropout(dropout_rate)(x2)
    transformer_features = tf.keras.layers.Add()([x1, x2])
    transformer_features = tf.keras.layers.LayerNormalization()(transformer_features)
    features = transformer_features

    # TAHE
    userHistoryRepresentation = CTHE(embedding_size)(
        [prevTimeRepresentations, curTimeRepresentation, recentTimestamps, features])
    Popular = CTHE(embedding_size)(
        [prevTimeRepresentations, hprevTimeRepresentations[:, 0, :], recentTimestamps, features])
    userHistoryRepresentation = (userHistoryRepresentation + Popular) / 2

    # combination
    x = concatenate([userVector, itemVector, curTimeRepresentation, userHistoryRepresentation])  # ,userfVector
    in_shape = embedding_size * 4

    for i in range(depth):
        if i == depth - 1:
            x = tf.keras.layers.Dense(1, input_shape=(in_shape,))(x)
        else:
            x = tf.keras.layers.Dense(width, input_shape=(in_shape,))(x)
            x = tf.keras.layers.Activation('relu')(x)
            if dropout_rate is not None:
                x = tf.keras.layers.Dropout(dropout_rate)(x)
        in_shape = width

    outputs = tf.keras.layers.Activation('sigmoid')(x)

    # model = tf.keras.Model(inputs=[userInput ,itemInput, monthInput, dayInput, dateInput, hourInput, curTimestampInput] + [recentMonthInput[i] for i in range(sequence_length)] + [recentDayInput[i] for i in range(sequence_length)] + [recentDateInput[i] for i in range(sequence_length)] + [recentHourInput[i] for i in range(sequence_length)] + [recentTimestampInput[i] for i in range(sequence_length)] + [recentItemidInput[i] for i in range(sequence_length)], outputs=outputs)

    model = tf.keras.Model(
        inputs=[userInput, itemInput, monthInput, dayInput, dateInput, hourInput, curTimestampInput] + [
            recentMonthInput[i] for i in range(sequence_length)] + [recentDayInput[i] for i in
                                                                    range(sequence_length)] + [recentDateInput[i] for i
                                                                                               in range(
                sequence_length)] + [recentHourInput[i] for i in range(sequence_length)] + [recentTimestampInput[i] for
                                                                                            i in
                                                                                            range(sequence_length)] + [
                   recentItemidInput[i] for i in range(sequence_length)] + [hotMonthInput[i] for i in range(popular_length)] + [
                   hotDayInput[i] for i in range(popular_length)] + [hotDateInput[i] for i in range(popular_length)] + [hotHourInput[i] for i in
                                                                                              range(popular_length)],
        outputs=outputs)
    return model
