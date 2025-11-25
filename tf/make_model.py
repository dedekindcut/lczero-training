#!/usr/bin/env python3
import argparse
import os
import yaml
import tfprocess

argparser = argparse.ArgumentParser(description="Convert net to model.")
argparser.add_argument("--start",
                       type=int,
                       default=0,
                       help="Offset to set global_step to.")
argparser.add_argument("--cfg",
                       type=argparse.FileType("r"),
                       help="yaml configuration with training parameters")
args = argparser.parse_args()
cfg = yaml.safe_load(args.cfg.read())
print(yaml.dump(cfg, default_flow_style=False))
START_FROM = args.start

tfp = tfprocess.TFProcess(cfg)
tfp.init_net()

# Explicitly restore the requested checkpoint
root_dir = os.path.join(cfg["training"]["path"], cfg["name"])
checkpoint_path = os.path.join(root_dir, cfg["name"] + "-" + str(START_FROM))

if os.path.exists(checkpoint_path + ".index"):
    print(f"Restoring weights from {checkpoint_path}...")
    # tfprocess stores the checkpoint object as self.ckpt, not self.checkpoint (based on my reading of TFProcess.__init__ in source, 
    # wait, the restore() method used self.checkpoint. Let's verify which one it is.
    # In TFProcess.__init__ (standard Lc0): self.ckpt = tf.train.Checkpoint(...) 
    # In TFProcess.restore(): self.checkpoint.restore(...)
    # It seems there is inconsistency or aliasing. Let's assume self.ckpt based on typical TF usage, 
    # but check if self.checkpoint exists.
    
    # Actually, let's look at the `restore` method I read: "self.checkpoint.restore"
    # So TFProcess likely has `self.checkpoint`.
    
    if hasattr(tfp, 'checkpoint'):
        tfp.checkpoint.restore(checkpoint_path).expect_partial()
    elif hasattr(tfp, 'ckpt'):
        tfp.ckpt.restore(checkpoint_path).expect_partial()
    else:
        print("Error: Could not find checkpoint object in TFProcess (tried .checkpoint and .ckpt)")
        exit(1)
    print("Restoration complete.")
else:
    print(f"Warning: Checkpoint {checkpoint_path} not found! Saving BASE model + random init.")

tfp.global_step.assign(START_FROM)

if not os.path.exists(root_dir):
    os.makedirs(root_dir)
# We don't need to save the checkpoint again, just export the weights.
# tfp.manager.save(checkpoint_number=START_FROM) 

print("Wrote model to {}".format(tfp.manager.latest_checkpoint))
path = os.path.join(tfp.root_dir, tfp.cfg["name"])
leela_path = path + "-" + str(START_FROM)
swa_path = path + "-swa-" + str(START_FROM)
tfp.net.pb.training_params.training_steps = START_FROM
tfp.save_leelaz_weights(leela_path)
if tfp.swa_enabled:
    tfp.save_swa_weights(swa_path)
