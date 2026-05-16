import { expect, test } from '@playwright/test'

test('phase 0 app boots and responds with worker status', async ({ page }) => {
  await page.goto('/')

  await expect(page.getByRole('heading', { name: 'WebGPU Style Transfer — Phase 0' })).toBeVisible()
  await expect(page.getByText(/Worker responded to ping/i)).toBeVisible()
  await expect(page.getByText(/WebGPU (device initialized|fallback)/i)).toBeVisible()
})
