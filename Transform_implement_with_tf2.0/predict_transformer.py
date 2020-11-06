def evaluate(inp_sentence):
start_token = [tokenizer_pt.vocab_size]
end_token = [tokenizer_pt.vocab_size + 1]

# 输入语句是葡萄牙语，增加开始和结束标记
inp_sentence = start_token + tokenizer_pt.encode(inp_sentence) + end_token
encoder_input = tf.expand_dims(inp_sentence, 0)

# 因为目标是英语，输入 transformer 的第一个词应该是
# 英语的开始标记。
decoder_input = [tokenizer_en.vocab_size]
output = tf.expand_dims(decoder_input, 0)
  
for i in range(MAX_LENGTH):
enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
    encoder_input, output)

# predictions.shape == (batch_size, seq_len, vocab_size)
predictions, attention_weights = transformer(encoder_input, 
                                             output,
                                             False,
                                             enc_padding_mask,
                                             combined_mask,
                                             dec_padding_mask)

# 从 seq_len 维度选择最后一个词
predictions = predictions[: ,-1:, :]  # (batch_size, 1, vocab_size)

predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

# 如果 predicted_id 等于结束标记，就返回结果
if predicted_id == tokenizer_en.vocab_size+1:
  return tf.squeeze(output, axis=0), attention_weights

# 连接 predicted_id 与输出，作为解码器的输入传递到解码器。
output = tf.concat([output, predicted_id], axis=-1)

return tf.squeeze(output, axis=0), attention_weights