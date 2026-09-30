// Read the same semantic colors used by both pages' CSS.
const styles = getComputedStyle(document.documentElement);
export const theme = Object.fromEntries(['page', 'surface', 'title', 'accent', 'accent-soft'].map(name => [name, styles.getPropertyValue('--' + name).trim()]));
