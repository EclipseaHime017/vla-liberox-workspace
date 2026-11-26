import { describe, expect, it } from "vitest";
import { densities, queryIndexAt } from "./ReturnReplay";

describe("return replay", () => {
  it("shows the query whose chunk is executing at each step", () => {
    const queries = [{ step: 0 }, { step: 10 }, { step: 15 }];
    expect([0, 9, 10, 14, 15, 40].map((step) => queryIndexAt(queries, step))).toEqual([0, 0, 1, 1, 2, 2]);
  });

  it("turns sample counts into densities over unequal bins", () => {
    expect(densities([2, 2], [0, 1, 3], 4)).toEqual([0.5, 0.25]);
  });
});
