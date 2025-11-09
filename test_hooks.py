#!/usr/bin/env python3
"""
Quick test to understand TransformerLens hook API
"""

import torch
from transformer_lens import HookedTransformer

print("Loading model...")
model = HookedTransformer.from_pretrained("gpt2-small", device="cpu")
print("Model loaded!\n")

# Test 1: Understanding add_hook return value
print("Test 1: Checking add_hook return value")
def test_hook(activation, hook):
    print(f"  Hook called on {hook.name}")
    return activation

handle = model.add_hook("blocks.0.attn.hook_z", test_hook)
print(f"  add_hook returned: {handle}")
print(f"  Type: {type(handle)}")

# Test 2: Run a forward pass to trigger hook
print("\nTest 2: Running forward pass with hook")
tokens = model.to_tokens("Hello world")
logits = model(tokens)
print("  Forward pass completed")

# Test 3: Remove hook
print("\nTest 3: Removing hook")
if handle is not None:
    print(f"  Attempting to remove hook...")
    try:
        handle.remove()
        print("  Hook removed successfully!")
    except Exception as e:
        print(f"  Error removing hook: {e}")
else:
    print("  ERROR: handle is None!")

# Test 4: Alternative hook context manager
print("\nTest 4: Using run_with_hooks")
def modulation_hook(activation, hook):
    return activation * 2.0

result = model.run_with_hooks(
    tokens,
    fwd_hooks=[("blocks.0.attn.hook_z", modulation_hook)]
)
print("  run_with_hooks completed successfully!")

# Test 5: Generate with hooks
print("\nTest 5: Generate with context manager")
model.reset_hooks()  # Clear any existing hooks

hook_handle = model.add_hook("blocks.0.attn.hook_z", modulation_hook)
if hook_handle:
    print(f"  Hook added: {hook_handle}")
    try:
        output = model.generate(tokens, max_new_tokens=5)
        print(f"  Generation successful: {model.to_string(output[0])}")
        hook_handle.remove()
        print("  Hook removed")
    except Exception as e:
        print(f"  Error during generation: {e}")
else:
    print("  ERROR: Could not add hook")

print("\nTest 6: Using hook context")
with model.hooks(fwd_hooks=[("blocks.0.attn.hook_z", modulation_hook)]):
    output = model.generate(tokens, max_new_tokens=5)
    print(f"  Generation with context: {model.to_string(output[0])}")
print("  Hooks automatically cleaned up")

print("\n✅ All tests complete!")
