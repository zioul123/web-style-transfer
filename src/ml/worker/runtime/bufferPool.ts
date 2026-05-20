export const reusableBufferPool: Map<string, GPUBuffer[]> = new Map()

export const getBufferPoolKey = (size: number, usage: number): string => `${size}:${usage}`

export const acquireReusableBuffer = (device: GPUDevice, size: number, usage: number): GPUBuffer => {
  const key = getBufferPoolKey(size, usage)
  const pool = reusableBufferPool.get(key)
  if (pool !== undefined && pool.length > 0) {
    const buffer = pool.pop()
    if (buffer !== undefined) return buffer
  }
  return device.createBuffer({ size, usage })
}

export const releaseReusableBuffer = (size: number, usage: number, buffer: GPUBuffer): void => {
  const key = getBufferPoolKey(size, usage)
  const pool = reusableBufferPool.get(key)
  if (pool === undefined) {
    reusableBufferPool.set(key, [buffer])
    return
  }
  pool.push(buffer)
}
