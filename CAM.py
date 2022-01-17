# TODO 수정수정수정


# gradcam reshape_transform for vit
def reshape_transform(tensor, height=7, width=7):
    result = tensor.reshape(tensor.size(0),
                            height, width, tensor.size(2))

    # like in CNNs.
    result = result.permute(0, 3, 1, 2)
    return result

model = Model(config)
model.load_state_dict(torch.load(f'{config.model.name}/default/version_0/checkpoints/best_loss.ckpt')['state_dict'])
model = model.cuda().eval()
config.val_loader.batch_size = 16
datamodule = PetfinderDataModule(train_df, val_df, config)
images, grayscale_cams, preds, labels = model.check_gradcam(
                                            datamodule.val_dataloader(),
                                            target_layer=model.backbone.layers[-1].blocks[-1].norm1,
                                            target_category=None,
                                            reshape_transform=reshape_transform)
plt.figure(figsize=(12, 12))
for it, (image, grayscale_cam, pred, label) in enumerate(zip(images, grayscale_cams, preds, labels)):
    plt.subplot(4, 4, it + 1)
    visualization = show_cam_on_image(image, grayscale_cam)
    plt.imshow(visualization)
    plt.title(f'pred: {pred:.1f} label: {label}')
    plt.axis('off')