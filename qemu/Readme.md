# References

1. install a ubuntu on qemu arm https://wiki.ubuntu.com/ARM64/QEMU
2. https://hackmd.io/@MarconiJiang/qemu_beginner
3. download qemu images https://en.wikibooks.org/wiki/QEMU/Images
4. https://wiki.archlinux.org/title/QEMU

```
sudo qemu-system-aarch64 \
-m 1024 -cpu cortex-a57 -M virt -nographic \
-pflash /usr/share/AAVMF/AAVMF_CODE.fd \
-pflash flash1.img \
-drive if=none,\
file=jammy-server-cloudimg-arm64.img,id=hd0 \
-device virtio-blk-device,drive=hd0 \
-netdev type=tap,id=net0 \
-device virtio-net-device,netdev=net0,mac=$randmac

sudo qemu-system-aarch64 \
-m 1024 -cpu cortex-a57 -M virt -nographic \
file=image_file,format=raw \
,id=hd0 \
-device virtio-blk-device,drive=hd0 \
-netdev type=tap,id=net0 \
-device virtio-net-device,netdev=net0,mac=$randmac
```


5. crate disk for qemu https://serverfault.com/questions/246835/convert-directory-to-qemu-kvm-virtual-disk-image
```
mount -t ext4 -o loop example.img /mnt/example

```

6. https://stackoverflow.com/questions/48989937/qemu-how-to-use-the-virt-board very same idea
