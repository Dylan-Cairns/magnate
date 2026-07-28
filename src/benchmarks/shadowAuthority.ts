export function legacyAuthoritativeValue<T>(
  legacyValue: T,
  candidateValue: T
): T {
  void candidateValue;
  return legacyValue;
}
