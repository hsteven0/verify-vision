const IMAGE_ROW_HEIGHT = 58;
const IMAGE_OVERSCAN = 5;
export const VIRTUALIZE_IMAGE_LIBRARY_AFTER = 80;

export function imageLibraryWindow(
  total: number,
  scrollTop: number,
  viewportHeight: number,
): { start: number; end: number } {
  if (total <= VIRTUALIZE_IMAGE_LIBRARY_AFTER) return { start: 0, end: total };
  const start = Math.max(0, Math.floor(scrollTop / IMAGE_ROW_HEIGHT) - IMAGE_OVERSCAN);
  const end = Math.min(total, Math.ceil((scrollTop + viewportHeight) / IMAGE_ROW_HEIGHT) + IMAGE_OVERSCAN);
  return { start, end };
}

export { IMAGE_ROW_HEIGHT };
