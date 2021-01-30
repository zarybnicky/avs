/**
 * @file    tree_mesh_builder.h
 * @brief   Parallel Marching Cubes implementation using OpenMP tasks + octree early elimination
 * @author  Jakub Zárybnický <xzaryb00@stud.fit.vutbr.cz>
 * @date    20.12.2020
 **/

#ifndef TREE_MESH_BUILDER_H
#define TREE_MESH_BUILDER_H

#include "base_mesh_builder.h"

class TreeMeshBuilder : public BaseMeshBuilder
{
public:
    TreeMeshBuilder(unsigned gridEdgeSize);

protected:
    unsigned marchCubes(const ParametricScalarField &field);
    unsigned octreeStep(const Vec3_t<float> &p, unsigned a, const ParametricScalarField &field);
    float evaluateFieldAt(const Vec3_t<float> &pos, const ParametricScalarField &field);
    void emitTriangle(const Triangle_t &triangle);
    const Triangle_t *getTrianglesArray() const;
    std::vector<Triangle_t> mTriangles;
};

#endif // TREE_MESH_BUILDER_H
