/**
 * @file    cached_mesh_builder.h
 * @brief   Parallel Marching Cubes implementation using pre-computed field
 * @author  Jakub Zárybnický <xzaryb00@stud.fit.vutbr.cz>
 * @date    20.12.2020
 **/

#ifndef CACHED_MESH_BUILDER_H
#define CACHED_MESH_BUILDER_H

#include <vector>
#include "base_mesh_builder.h"

class CachedMeshBuilder : public BaseMeshBuilder
{
public:
  CachedMeshBuilder(unsigned gridEdgeSize);

protected:
  unsigned marchCubes(const ParametricScalarField &field);
  float evaluateFieldAt(const Vec3_t<float> &pos, const ParametricScalarField &field);
  void emitTriangle(const Triangle_t &triangle);
  const Triangle_t *getTrianglesArray() const;
  std::vector<std::vector<Triangle_t> > *mTriangles;
  std::vector<Triangle_t> outVector;
  std::vector<float> mCache;
};

#endif // CACHED_MESH_BUILDER_H
