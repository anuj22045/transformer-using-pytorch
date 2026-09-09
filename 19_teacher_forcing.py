"""
WHAT IS TEACHER FORCING?

Teacher Forcing is a training technique where we give the Decoder the correct previous target token instead of its own previous predicted token.
Suppose we want:
English:
I love cats

French:
Je t'aime les chats
During training, the Decoder receives:

<START> Je t'aime les
and learns to predict: Je t'aime les chats <END>

So:
Decoder Input: <START> → Je → t'aime → les

Target:Je → t'aime → les → chats

The correct previous words are given to the Decoder.
That's Teacher Forcing.

-----------------------------------------------------------------------------------------------------------------------------------
HOW DOES IT WORK?

Let's take:
Target sentence:
Je t'aime les chats
Add special tokens:

<START> Je t'aime les chats <END>
Now we create two sequences.
Decoder Input
<START> Je t'aime les chats
Target
Je t'aime les chats <END>
Notice that the target is shifted by one position.

Decoder Input          Target

<START>          →     Je
Je               →     t'aime
t'aime            →     les
les               →     chats
chats             →     <END>
This is the core idea of Teacher Forcing.
"""

import torch


# ---------------------------------------------------------
# Target Sentence
# ---------------------------------------------------------

# Token IDs:
#
# 1 = <START>
# 2 = Je
# 3 = t'aime
# 4 = les
# 5 = chats
# 6 = <END>

target = torch.tensor([
    [1, 2, 3, 4, 5, 6]
])


print("Original Target:")
print(target)

print("\nOriginal Target Shape:")
print(target.shape)


# ---------------------------------------------------------
# Teacher Forcing
# ---------------------------------------------------------

decoder_input = target[:, :-1]

target_output = target[:, 1:]


print("\nDecoder Input:")
print(decoder_input)

print("\nDecoder Input Shape:")
print(decoder_input.shape)


print("\nTarget Output:")
print(target_output)

print("\nTarget Output Shape:")
print(target_output.shape)