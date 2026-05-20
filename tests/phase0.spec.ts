import { expect, test } from '@playwright/test'

test('phase 0 app boots and responds with worker status', async ({ page }) => {
  await page.goto('/')

  await expect(page.getByRole('heading', { name: /WebGPU Style Transfer/i })).toBeVisible()
  await expect(page.getByText(/Worker ping OK/i)).toBeVisible()
  await expect(page.getByText(/WebGPU/i).first()).toBeVisible()
})
